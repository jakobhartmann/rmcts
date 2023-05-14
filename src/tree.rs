#![allow(unused)]

use crate::eg_env::{Ckpt, EgraphEnv};
// use crate::env::Env;
use crate::node::{Node, NodeStub, print_tree};
use crate::pool_manager;
use crate::workers::Reply;
use crate::utils::save_data_to_file;

#[allow(unused_imports)]
use egg::{Analysis, CostFunction, EGraph, Id, Language, LpCostFunction, RecExpr, Rewrite, Extractor};
use rand::Rng;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::rc::Rc;
use std::thread;
use std::time::{Duration, Instant};
use std::path::PathBuf;
use serde_json::json;
// use log::info;

pub struct ExpTask<L, N>
where
    L: Language + 'static + egg::FromOp + std::marker::Send + std::fmt::Display,
    N: Analysis<L> + Clone + 'static + std::default::Default + std::marker::Send,
    N::Data: Clone,
    <N as Analysis<L>>::Data: Send,
{
    // pub checkpoint_data: Vec<usize>,
    pub checkpoint_data: Ckpt<L, N>,
    pub shallow_copy_node: NodeStub,
    d1: PhantomData<L>,
    d2: PhantomData<N>,
}

#[derive(Clone)]
pub struct SimTask<L, N>
where
    L: Language + 'static + egg::FromOp + std::marker::Send + std::fmt::Display,
    N: Analysis<L> + Clone + 'static + std::default::Default + std::marker::Send,
    N::Data: Clone,
    <N as Analysis<L>>::Data: Send,
{
    pub checkpoint_data: Ckpt<L, N>,
    pub action: usize,
    pub saving_idx: u32,
    pub action_applied: bool,
    pub child_saturated: bool,
    pub children_saturated: Vec<bool>,
    pub children_saturated_cnt: usize,
    d1: PhantomData<L>,
    d2: PhantomData<N>,
}

pub struct Tree<L, N, CF>
where
    L: Language + 'static + egg::FromOp + std::marker::Send + std::fmt::Display,
    N: Analysis<L> + Clone + 'static + std::default::Default + std::marker::Send,
    N::Data: Clone,
    <N as Analysis<L>>::Data: Send,
    CF: CostFunction<L> + LpCostFunction<L, N> + Clone + std::marker::Send + 'static,
    usize: From<<CF as CostFunction<L>>::Cost>,
{
    // from param
    budget: u32,
    gamma: f32,

    // data and concurrency
    exp_pool: pool_manager::PoolManager<L, N, CF>,
    sim_pool: pool_manager::PoolManager<L, N, CF>,
    // ckpts: HashMap<u32, Vec<usize>>,
    ckpts: HashMap<u32, Ckpt<L, N>>,
    ckpts_old: HashMap<u32, Ckpt<L, N>>,
    cf: CF,
    lp_extract: bool,

    prune_actions: bool,
    // rollout_strategy: String,
    subtree_caching: bool,
    select_max_uct_action: bool,

    // experiment tracking
    output_dir: PathBuf,

    // for planning
    root_node: Rc<RefCell<Node>>,
    best_child_node: Option<Rc<RefCell<Node>>>,
    global_saving_idx: u32,
    simulation_count: u32,
    expansion_tasks: HashMap<u32, ExpTask<L, N>>,
    expansion_nodes_copy: HashMap<u32, Rc<RefCell<Node>>>,
    simulation_tasks: HashMap<u32, SimTask<L, N>>,
    simulation_nodes_copy: HashMap<u32, Rc<RefCell<Node>>>,
    pending_expansion_tasks: VecDeque<u32>,
    pending_simulation_tasks: VecDeque<u32>,

    d1: PhantomData<L>,
    d2: PhantomData<N>,

    // egg
    node_limit: usize,
    time_limit: usize,
}

impl<L, N, CF> Tree<L, N, CF>
where
    L: Language + 'static + egg::FromOp + std::marker::Send + std::fmt::Display,
    N: Analysis<L> + Clone + 'static + std::default::Default + std::marker::Send,
    N::Data: Clone,
    <N as Analysis<L>>::Data: Send,
    CF: CostFunction<L> + LpCostFunction<L, N> + Clone + std::marker::Send + 'static,
    usize: From<<CF as CostFunction<L>>::Cost>,
{
    pub fn new(
        // mcts
        budget: u32,
        max_sim_step: u32,
        gamma: f32,
        expansion_worker_num: usize,
        simulation_worker_num: usize,
        prune_actions: bool,
        rollout_strategy: String,
        subtree_caching: bool,
        select_max_uct_action: bool,
        // experiment tracking
        output_dir: PathBuf,
        // egg
        egraph: EGraph<L, N>,
        id: Id,
        rules: Vec<Rewrite<L, N>>,
        cf: CF,
        lp_extract: bool,
        node_limit: usize,
        time_limit: usize,
    ) -> Self {
        assert_eq!(expansion_worker_num, 1); // more than 1 expansion may have problem
        Tree {
            budget: budget,
            gamma: gamma,

            exp_pool: pool_manager::PoolManager::new(
                "expansion",
                expansion_worker_num,
                gamma,
                max_sim_step,
                false,
                egraph.clone(),
                id.clone(),
                rules.clone(),
                cf.clone(),
                lp_extract,
                prune_actions,
                rollout_strategy.clone(),
                node_limit,
                time_limit,
            ),
            sim_pool: pool_manager::PoolManager::new(
                "simulation",
                simulation_worker_num,
                gamma,
                max_sim_step,
                false,
                egraph.clone(),
                id.clone(),
                rules.clone(),
                cf.clone(),
                lp_extract,
                prune_actions,
                rollout_strategy.clone(),
                node_limit,
                time_limit,
            ),
            ckpts: HashMap::new(),
            ckpts_old: HashMap::new(),
            cf: cf,
            lp_extract: lp_extract,

            prune_actions: prune_actions,
            // rollout_strategy: rollout_strategy,
            subtree_caching: subtree_caching,
            select_max_uct_action: select_max_uct_action,

            output_dir: output_dir,

            root_node: Node::dummy(),
            best_child_node: None,
            global_saving_idx: 0,
            simulation_count: 0,
            expansion_tasks: HashMap::new(),
            expansion_nodes_copy: HashMap::new(),
            simulation_tasks: HashMap::new(),
            simulation_nodes_copy: HashMap::new(),
            pending_expansion_tasks: VecDeque::new(),
            pending_simulation_tasks: VecDeque::new(),
            d1: PhantomData,
            d2: PhantomData,
            node_limit: node_limit,
            time_limit: time_limit,
        }
    }

    pub fn run_loop(
        &mut self,
        egraph: EGraph<L, N>,
        id: Id,
        rules: Vec<Rewrite<L, N>>,
        cost_threshold: usize,
        iter_limit: usize,
    ) -> EGraph<L, N> {
        // env
        // let mut env = Env::new(expr, rules, self.node_limit, self.time_limit);
        let mut env = EgraphEnv::new(
            egraph,
            id,
            rules,
            self.cf.clone(),
            self.lp_extract,
            self.prune_actions,
            self.node_limit,
            self.time_limit,
        );
        env.reset();

        // loop var
        let mut state = ();
        let mut reward;
        let mut done;
        let mut info;
        let mut iter = 0;
        let mut episode_reward = 0.0;
        let mut total_planning_time = 0;

        // env loop
        loop {
            let planning_time = Instant::now();
            let (action, pruned_root_actions, cached_subtree_nodes) = self.plan(&state, &mut env);
            let planning_time = planning_time.elapsed().as_secs();
            total_planning_time += planning_time;

            match action {
                usize::MAX => {
                    println!("Search stopped, because the egraph is fully saturated. Performing dummy final action.");
                    (state, reward, done, info) = env.step(0);
                    done = true;
                },
                _ => {
                    (state, reward, done, info) = env.step(action);
                }
            }

            iter += 1;
            episode_reward += reward;

            println!(
                "Iter {}; planning time {}s; reward {}; episode_reward {}; best cost {}",
                iter, planning_time, reward, episode_reward, info.best_cost
            );
            println!("{}", info.report);
            println!("************************");

            let iteration = json!({
                // iteration data
                "action": action,
                "planning_time": planning_time,
                "total_planning_time": total_planning_time,
                "reward": reward,
                "episode_reward": episode_reward,
                "base_cost": env.base_cost,
                "best_cost": info.best_cost,
                "done": done,
                "pruned_root_actions": pruned_root_actions,
                "cached_subtree_nodes": cached_subtree_nodes,

                // runner stats
                "runner_iterations": info.report.iterations,
                "runner_stop_reason": info.report.stop_reason,
                "runner_egraph_nodes": info.report.egraph_nodes,
                "runner_egraph_classes": info.report.egraph_classes,
                "runner_memo_size": info.report.memo_size,
                "runner_rebuilds": info.report.rebuilds,
                "runner_total_time": info.report.total_time,
                "runner_search_time": info.report.search_time,
                "runner_apply_time": info.report.apply_time,
                "runner_rebuild_time": info.report.rebuild_time,
            });

            save_data_to_file(&iteration, &self.output_dir, "rmcts_iteration_data.txt");

            if done || info.best_cost < cost_threshold || iter >= iter_limit {
                break;
            }
        }

        // Extract best cost and expression
        let extractor = Extractor::new(&env.egraph, self.cf.clone());
        let (rmcts_cost, rmcts_expr) = extractor.find_best(env.root_id);
        save_data_to_file(&rmcts_expr, &self.output_dir, "rmcts_expr.txt");

        println!(
            "[RMCTS] Done:: base_cost {} -> cost {:?} with iter {} and time {}s",
            env.base_cost, info.best_cost, iter, total_planning_time,
        );

        // Save RMCTS stats
        let rmcts_stats = json!({
            "base_cost": env.base_cost,
            "best_cost": info.best_cost,
            "optimization_time": total_planning_time,
        });
        save_data_to_file(&rmcts_stats, &self.output_dir, "rmcts_stats.txt");

        self.close();
        env.egraph
    }

    // fn plan(&mut self, _state: &(), env: &Env<L, N>) -> usize {
    // Returns: (action, #pruned root actions, #cached nodes)
    fn plan(&mut self, _state: &(), env: &mut EgraphEnv<L, N, CF>) -> (usize, usize, u32) {
        // skip if action space is 1
        let action_n = env.get_action_space();
        if action_n == 1 {
            return (0, 0, 0);
        }

        // clear
        self.global_saving_idx = 0;
        self.simulation_count = 0;
        self.ckpts.clear();
        self.expansion_tasks.clear();
        self.expansion_nodes_copy.clear();
        self.simulation_tasks.clear();
        self.simulation_nodes_copy.clear();
        self.pending_expansion_tasks.clear();
        self.pending_simulation_tasks.clear();
        self.exp_pool.wait_until_all_idle();
        self.sim_pool.wait_until_all_idle();

        match &self.best_child_node {
            Some(best_child_node) => {
                println!("Reuse cached subtree with {} nodes.", best_child_node.borrow().visit_count);
                println!("Cached root node has {} pruned children.", best_child_node.borrow().children_pruned);
                self.root_node = best_child_node.clone();
                self.root_node.borrow_mut().parent = None;
                self.root_node.borrow_mut().is_head = true;

                // Update checkpoints
                let mut node_stack = vec![self.root_node.clone()];
                while !node_stack.is_empty() {
                    let node = node_stack.pop().unwrap();

                    match self.ckpts_old.get(&node.borrow().checkpoint_idx) {
                        Some(checkpoint) => {
                            self.ckpts.insert(self.global_saving_idx, checkpoint.clone());
                        },
                        // Checkpoint can be missing, if the expansion returned done = true
                        None => {}
                    }

                    node.borrow_mut().checkpoint_idx = self.global_saving_idx;
                    self.global_saving_idx += 1;

                    for child_node in node.borrow().children.clone().into_iter() {
                        if child_node.is_some() {
                            node_stack.push(child_node.clone().unwrap());
                        }
                    };
                };
            },
            None => {
                println!("No cached subtree.");
                // Perform action pruning for root node
                let (children_saturated, children_saturated_cnt) = env.action_pruning(vec![false; action_n]);
                self.root_node = Node::new(None, action_n, self.global_saving_idx, self.gamma, true, None, children_saturated, children_saturated_cnt);
                println!("Pruned {} children of the root node.", self.root_node.borrow().children_saturated_cnt);

                // build current state
                self.ckpts.insert(self.global_saving_idx, env.checkpoint());
                self.global_saving_idx += 1;
            }
        }

        let root_visit_count = self.root_node.borrow().visit_count;
        self.simulation_count = self.root_node.borrow().visit_count;

        // run main mcts
        let mut depth = 0;
        let mut root_saturated = false;
        for sim_idx in root_visit_count..self.budget {
            // If all children of the root saturated, stop immediately. There is no point in continuing the search.
            if self.root_node.borrow().all_child_saturated() {
                root_saturated = true;
                break
            }

            // print_tree(&self.root_node.borrow(), 3, 0);

            let (d, saturated) = self.simulate_single_step(sim_idx);
            depth = std::cmp::max(depth, d);

            // If all leaf nodes are saturated, stop the search
            if saturated {
                break;
            }
        }

        if root_saturated {
            return (usize::MAX, self.root_node.borrow().children_pruned, root_visit_count)
        }

        // clean up
        println!(
            "complete count {}/{} - max_depth {}",
            self.simulation_count, self.budget, depth
        );
        thread::sleep(Duration::from_secs(1));

        // it is a bad idea to termiante a thread, perhaps just timeout a function in worker
        // thread, as a way to handle stragger

        // final action
        let best_action;
        if self.select_max_uct_action {
            best_action = self.root_node.borrow().select_uct_action(true)
        } else {
            best_action = self.root_node.borrow().select_max_visited_action()
        }

        // Subtree caching
        if self.subtree_caching {
            self.best_child_node = self.root_node.borrow().children[best_action].clone();
            self.ckpts_old = self.ckpts.clone();
        }

        (best_action, self.root_node.borrow().children_pruned, root_visit_count)
    }

    fn simulate_single_step(&mut self, sim_idx: u32) -> (u32, bool) {
        // Selection
        let mut curr_node: Rc<RefCell<Node>> = Rc::clone(&self.root_node);
        let mut curr_depth = 1;
        let mut rng = rand::thread_rng();
        let need_expansion;

        loop {
            let rand = rng.gen_range(0.0..1.0);
            if ((!curr_node.borrow().all_child_visited()) && (!curr_node.borrow().all_non_pruned_child_visited()))
                && (curr_node.borrow().no_child_available()
                || curr_node.borrow().is_head
                || !curr_node.borrow().is_head && rand < 0.5)
            // if (curr_node.borrow().no_child_available() && (!curr_node.borrow().all_child_visited()))
            //     || (curr_node.borrow().is_head && (!curr_node.borrow().all_child_visited()))
            //     || ((!curr_node.borrow().is_head && !curr_node.borrow().all_child_visited())
            //         && rand < 0.5)
            {
                // Don't expand if all children (that are not pruned) have already been visited.
                // If no child node has been updated, we have to expand anyway.
                // Or if root node is not fully visited.
                // Or if non-root node is not fully visited and {with prob 1/2}.

                let cloned_curr_node = curr_node.borrow().shallow_clone();
                let checkpoint_data = self
                    .ckpts
                    .get(&curr_node.borrow().checkpoint_idx)
                    .unwrap()
                    .clone();
                // println!("{:?}", curr_node.children);

                // Record the task
                self.expansion_tasks.insert(
                    sim_idx,
                    ExpTask {
                        checkpoint_data: checkpoint_data,
                        shallow_copy_node: cloned_curr_node,
                        d1: PhantomData,
                        d2: PhantomData,
                    },
                );
                self.expansion_nodes_copy
                    .insert(sim_idx, Rc::clone(&curr_node));
                self.pending_expansion_tasks.push_back(sim_idx);

                need_expansion = true;
                break;
            }

            // If all leaf nodes are saturated, stop the search
            if curr_node.borrow().all_child_saturated() {
                println!("All children saturated!");
                return (curr_depth, true)
            }
            let action = curr_node.borrow().select_uct_action(false);
            let reward = curr_node.borrow().rewards[action].clone();
            curr_node
                .borrow_mut()
                .update_history(sim_idx, action, reward);

            if curr_node.borrow().dones[action] {
                // exceed maximum depth
                need_expansion = false;
                break;
            }

            // one-level deeper
            curr_depth += 1;
            let child: Rc<RefCell<Node>> = curr_node.borrow().children[action]
                .as_ref()
                .unwrap()
                .clone();
            curr_node = child;
        }

        // Expansion
        if need_expansion {
            // schedule
            while !self.pending_expansion_tasks.is_empty() && self.exp_pool.has_idle_server() {
                let task_idx = self.pending_expansion_tasks.pop_front().unwrap();
                let exp_task = self.expansion_tasks.remove(&task_idx).unwrap(); // remove get
                                                                                // ownership
                self.exp_pool
                    .assign_expansion_task(exp_task, self.global_saving_idx, task_idx);
                self.global_saving_idx += 1;
            }
            // update
            if self.exp_pool.occupancy() > 0.99 {
                let reply = self.exp_pool.get_complete_task();
                if let Reply::DoneExpansion(
                    expand_action,
                    _next_state,
                    reward,
                    done,
                    child_saturated,
                    children_saturated,
                    children_saturated_cnt,
                    new_checkpoint_data,
                    saving_idx,
                    task_idx,
                ) = reply
                {
                    let curr_node_copy = self.expansion_nodes_copy.remove(&task_idx).unwrap();
                    curr_node_copy
                        .borrow_mut()
                        .update_history(task_idx, expand_action, reward);
                    curr_node_copy.borrow_mut().dones[expand_action] = done;
                    curr_node_copy.borrow_mut().rewards[expand_action] = reward;

                    if done {
                        // If this expansion result in a terminal node,
                        // perform update directly (simulation is not needed)
                        assert!(new_checkpoint_data.is_none());
                        curr_node_copy.borrow_mut().add_child(
                            expand_action,
                            saving_idx,
                            self.gamma,
                            child_saturated,
                            children_saturated,
                            children_saturated_cnt,
                            Rc::clone(&curr_node_copy),
                        );
                        self.incomplete_update(Rc::clone(&curr_node_copy), task_idx);
                        self.complete_update(Rc::clone(&curr_node_copy), task_idx, 0.0);
                        self.simulation_count += 1;
                    } else {
                        // ELSE add_child will be done after simulation!
                        // Add task to pending simulation
                        assert!(new_checkpoint_data.is_some());
                        let new_checkpoint_data = new_checkpoint_data.unwrap();
                        self.ckpts.insert(saving_idx, new_checkpoint_data.clone());
                        self.simulation_tasks.insert(
                            task_idx,
                            SimTask {
                                checkpoint_data: new_checkpoint_data,
                                action: expand_action,
                                saving_idx: saving_idx,
                                action_applied: true,
                                child_saturated: child_saturated,
                                children_saturated: children_saturated,
                                children_saturated_cnt: children_saturated_cnt,
                                d1: PhantomData,
                                d2: PhantomData,
                            },
                        );
                        self.simulation_nodes_copy
                            .insert(task_idx, Rc::clone(&curr_node_copy));
                        self.pending_simulation_tasks.push_back(task_idx)
                    }
                } else {
                    panic!("DoneExpansion destructure fails");
                }
            }
        } else {
            // no need expansion
            // reach terminal node
            self.incomplete_update(Rc::clone(&curr_node), sim_idx);
            self.complete_update(Rc::clone(&curr_node), sim_idx, 0.0);
            self.simulation_count += 1;
        }

        // Simulation
        // schedule
        while !self.pending_simulation_tasks.is_empty() && self.sim_pool.has_idle_server() {
            // pop a task
            let task_idx = self.pending_simulation_tasks.pop_front().unwrap();
            let sim_task = self.simulation_tasks.get(&task_idx).unwrap().clone();
            let curr_node_copy = Rc::clone(self.simulation_nodes_copy.get(&task_idx).unwrap());
            // schedule
            self.sim_pool.assign_simulation_task(sim_task, task_idx);
            // incomplete update
            self.incomplete_update(Rc::clone(&curr_node_copy), task_idx);
        }
        // update
        while self.sim_pool.occupancy() > 0.5
            || (self.budget == sim_idx + 1 && self.simulation_count != self.budget)
        {
            let reply = self.sim_pool.get_complete_task();
            if let Reply::DoneSimulation(task_idx, accu_reward) = reply {
                // fetch
                let sim_task = self.simulation_tasks.remove(&task_idx).unwrap();
                let curr_node_copy = self.simulation_nodes_copy.remove(&task_idx).unwrap();
                assert!(sim_task.action_applied);
                // add-child
                curr_node_copy.borrow_mut().add_child(
                    sim_task.action,
                    sim_task.saving_idx,
                    self.gamma,
                    sim_task.child_saturated,
                    sim_task.children_saturated,
                    sim_task.children_saturated_cnt,
                    Rc::clone(&curr_node_copy),
                );
                self.complete_update(Rc::clone(&curr_node_copy), task_idx, accu_reward);
                self.simulation_count += 1;
            } else {
                panic!("DoneSimulation destructure fails");
            }
        }
        (curr_depth, false)
    }

    fn incomplete_update(&mut self, mut curr_node: Rc<RefCell<Node>>, idx: u32) {
        while !curr_node.borrow().is_head {
            curr_node.borrow_mut().update_incomplete(idx);
            let parent: Rc<RefCell<Node>> = Rc::clone(curr_node.borrow().parent.as_ref().unwrap());
            curr_node = parent;
        }
        self.root_node.borrow_mut().update_incomplete(idx);
    }

    fn complete_update(&mut self, mut curr_node: Rc<RefCell<Node>>, idx: u32, accu_reward: f32) {
        let mut rolling_accu_reward = accu_reward;
        while !curr_node.borrow().is_head {
            rolling_accu_reward = curr_node
                .borrow_mut()
                .update_complete(idx, rolling_accu_reward);
            let parent: Rc<RefCell<Node>> = Rc::clone(curr_node.borrow().parent.as_ref().unwrap());
            curr_node = parent;
        }
        self.root_node
            .borrow_mut()
            .update_complete(idx, rolling_accu_reward);
    }

    // fn update_saturated_children(&mut self, mut curr_node: Rc<RefCell<Node>>) {
    //     while curr_node.borrow().children_saturated_cnt == curr_node.borrow().action_n {
    //         let saturated_action = curr_node.borrow().action;
    //         if curr_node.borrow().is_head {
    //             break;
    //         }
    //         let parent: Rc<RefCell<Node>> = Rc::clone(curr_node.borrow().parent.as_ref().unwrap());
    //         curr_node = parent;
    //         curr_node.borrow_mut().children_saturated[saturated_action.unwrap()] = true;
    //         curr_node.borrow_mut().children_saturated_cnt += 1;
    //     }
    // }

    fn close(&mut self) {
        self.exp_pool.close();
        self.sim_pool.close();
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_if_map_take_ownership() {
        let a = vec![Some(1), None, Some(3)];
        let mut children: Vec<u32> = a.iter().map(|x| if x.is_some() { 1 } else { 0 }).collect();
        for (_i, j) in children.iter_mut().enumerate() {
            *j += 1;
        }
        for (i, j) in children.iter_mut().enumerate() {
            println!("{} - {}", i, j);
        }
    }

    #[test]
    fn test_rand() {
        let mut rng = rand::thread_rng();
        for _ in 0..5 {
            println!("rand gen {} ", rng.gen_range(0..10));
        }
    }
}
