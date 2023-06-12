#![allow(unused_imports)]

use std::path::PathBuf;
use egg::{Analysis, CostFunction, EGraph, Id, Language, RecExpr, Rewrite};
use crate::tree;

pub struct MCTSArgs {
    // mcts
    pub budget: u32,
    pub max_sim_step: u32,
    pub gamma: f32,
    pub expansion_worker_num: usize,
    pub simulation_worker_num: usize,
    pub lp_extract: bool,
    pub cost_threshold: usize,
    pub iter_limit: usize,
    pub prune_actions: bool,
    pub rollout_strategy: String,
    pub subtree_caching: bool,
    pub select_max_uct_action: bool,
    // experiment tracking
    pub output_dir: PathBuf,
    // egg
    pub node_limit: usize,
    pub time_limit: usize,
}

pub fn run_mcts<L, N, CF>(
    egraph: EGraph<L, N>,
    id: Id,
    rules: Vec<Rewrite<L, N>>,
    cf: CF,
    args: Option<MCTSArgs>,
) -> EGraph<L, N>
where
    L: Language + 'static + egg::FromOp + std::marker::Send + std::fmt::Display,
    N: Analysis<L> + Clone + 'static + std::default::Default + std::marker::Send,
    N::Data: Clone,
    <N as Analysis<L>>::Data: Send,
    CF: CostFunction<L> + Clone + std::marker::Send + 'static,
    usize: From<<CF as CostFunction<L>>::Cost>,
{
    // Args
    // mcts
    let mut budget = 12;
    let mut max_sim_step = 5;
    let mut gamma = 0.99;
    let mut expansion_worker_num = 1;
    let mut simulation_worker_num = 1;
    let mut lp_extract = false;
    let mut cost_threshold = 1;
    let mut iter_limit = 30;
    let mut prune_actions = false;
    let mut rollout_strategy = String::from("random");
    let mut subtree_caching = false;
    let mut select_max_uct_action = true; // true -> max uct action; false -> max visited action
    // let verbose = false;
    // experiment tracking
    let mut output_dir = PathBuf::new();
    // egg
    let mut node_limit = 10_000;
    let mut time_limit = 1;

    // overwrite params if possible
    match args {
        None => (),
        Some(args) => {
            // mcts
            budget = args.budget;
            max_sim_step = args.max_sim_step;
            gamma = args.gamma;
            expansion_worker_num = args.expansion_worker_num;
            simulation_worker_num = args.simulation_worker_num;
            lp_extract = args.lp_extract;
            cost_threshold = args.cost_threshold;
            iter_limit = args.iter_limit;
            prune_actions = args.prune_actions;
            rollout_strategy = args.rollout_strategy;
            subtree_caching = args.subtree_caching;
            select_max_uct_action = args.select_max_uct_action;
            // experiment tracking
            output_dir = args.output_dir;
            // egg
            node_limit = args.node_limit;
            time_limit = args.time_limit;
        }
    }

    // Run
    let mut mcts = tree::Tree::new(
        // mcts
        budget,
        max_sim_step,
        gamma,
        expansion_worker_num,
        simulation_worker_num,
        prune_actions,
        rollout_strategy,
        subtree_caching,
        select_max_uct_action,
        // experiment tracking
        output_dir,
        // egg
        egraph.clone(),
        id.clone(),
        rules.clone(),
        cf,
        lp_extract,
        node_limit,
        time_limit,
    );
    mcts.run_loop(egraph, id, rules.clone(), cost_threshold, iter_limit)
}
