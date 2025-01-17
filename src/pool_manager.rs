use crate::tree::{ExpTask, SimTask};
use crate::workers::{worker_loop, Message, Reply};

#[allow(unused_imports)]
use egg::{Analysis, CostFunction, EGraph, Id, Language, RecExpr, Rewrite};
use std::marker::PhantomData;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Status {
    Busy,
    Idle,
}

pub struct PoolManager<L, N, CF>
where
    L: Language + 'static + egg::FromOp + std::marker::Send + std::fmt::Display,
    N: Analysis<L> + Clone + 'static + std::default::Default + std::marker::Send,
    N::Data: Clone,
    <N as Analysis<L>>::Data: Send,
    CF: CostFunction<L> + Clone + std::marker::Send + 'static,
    usize: From<<CF as CostFunction<L>>::Cost>,
{
    #[allow(unused_variables, dead_code)]
    name: &'static str,
    work_num: usize,

    // self
    workers: Vec<thread::JoinHandle<()>>,
    worker_status: Vec<Status>,
    txs: Vec<Sender<Message<L, N>>>,
    rxs: Vec<Receiver<Reply<L, N>>>,
    d: PhantomData<CF>,
}

impl<L, N, CF> PoolManager<L, N, CF>
where
    L: Language + 'static + egg::FromOp + std::marker::Send + std::fmt::Display,
    N: Analysis<L> + Clone + 'static + std::default::Default + std::marker::Send,
    N::Data: Clone,
    <N as Analysis<L>>::Data: Send,
    CF: CostFunction<L> + Clone + std::marker::Send + 'static,
    usize: From<<CF as CostFunction<L>>::Cost>,
{
    pub fn new(
        // mcts
        name: &'static str,
        work_num: usize,
        gamma: f32,
        max_sim_step: u32,
        verbose: bool,
        egraph: EGraph<L, N>,
        id: Id,
        rules: Vec<Rewrite<L, N>>,
        cf: CF,
        lp_extract: bool,
        prune_actions: bool,
        rollout_strategy: String,
        // egg
        node_limit: usize,
        time_limit: usize,
    ) -> Self {
        // build workers
        let mut workers = Vec::new();
        let mut txs = Vec::new();
        let mut rxs = Vec::new();
        for i in 0..work_num {
            let (w, tx, rx) = worker_loop(
                i,
                gamma,
                max_sim_step,
                verbose,
                egraph.clone(),
                id.clone(),
                rules.clone(),
                cf.clone(),
                lp_extract,
                prune_actions,
                rollout_strategy.clone(),
                node_limit,
                time_limit,
            );
            workers.push(w);
            txs.push(tx);
            rxs.push(rx);
        }

        PoolManager {
            name: name,
            work_num: work_num,
            workers: workers,
            worker_status: vec![Status::Idle; work_num],
            txs: txs,
            rxs: rxs,
            d: PhantomData,
        }
    }

    pub fn has_idle_server(&mut self) -> bool {
        self.worker_status.contains(&Status::Idle)
    }

    pub fn assign_expansion_task(
        &mut self,
        exp_task: ExpTask<L, N>,
        global_saving_idx: u32,
        task_idx: u32,
    ) {
        let id = self.find_idle_worker();
        self.txs[id]
            .send(Message::Expansion(exp_task, global_saving_idx, task_idx))
            .unwrap();
    }

    pub fn assign_simulation_task(&mut self, sim_task: SimTask<L, N>, task_idx: u32) {
        let id = self.find_idle_worker();
        self.txs[id]
            .send(Message::Simulation(sim_task, task_idx))
            .unwrap();
    }

    #[allow(dead_code)]
    pub fn assign_nothing_task(&mut self) {
        let id = self.find_idle_worker();
        self.txs[id].send(Message::Nothing).unwrap();
    }

    fn find_idle_worker(&mut self) -> usize {
        for (i, status) in self.worker_status.iter_mut().enumerate() {
            match status {
                Status::Busy => (),
                Status::Idle => {
                    self.worker_status[i] = Status::Busy;
                    return i;
                }
            }
        }
        unreachable!("no idle worker");
    }

    pub fn occupancy(&mut self) -> f32 {
        (self
            .worker_status
            .iter()
            .fold(0, |acc, x| if x == &Status::Busy { acc + 1 } else { acc }) as f32)
            / (self.work_num as f32)
    }

    pub fn get_complete_task(&mut self) -> Reply<L, N> {
        loop {
            for i in 0..self.work_num {
                let reply = self.rxs[i].try_recv(); // non-blocking
                match reply {
                    Err(_) => (),
                    Ok(r) => {
                        self.worker_status[i] = Status::Idle;
                        return r;
                    }
                }
            }
        }
    }

    pub fn wait_until_all_idle(&mut self) {
        for id in 0..self.work_num {
            match self.worker_status[id] {
                Status::Idle => (),
                Status::Busy => {
                    self.rxs[id].recv().unwrap(); // block until workers finish
                    self.worker_status[id] = Status::Idle;
                }
            }
        }
    }

    pub fn close(&mut self) {
        // wait until all exit
        for id in 0..self.work_num {
            match self.worker_status[id] {
                Status::Idle => self.txs[id].send(Message::Exit).unwrap(),
                Status::Busy => {
                    self.rxs[id].recv().unwrap(); // block until workers finish
                    self.txs[id].send(Message::Exit).unwrap();
                    self.worker_status[id] = Status::Idle;
                }
            }
        }
        // join
        while self.workers.len() > 0 {
            let w = self.workers.remove(0); // get ownership
            w.join().unwrap();
        }
    }
}

// #[cfg(test)]
// mod test {
//     #![allow(unused_imports)]
//     // use super::{PoolManager, worker_loop};
//     use super::*;
//     // use std::sync::atomic::{AtomicUsize, Ordering};
//     // use std::sync::mpsc::{channel, sync_channel};
//     // use std::sync::{Arc, Barrier};
//     use egg::*;
//     use std::thread::sleep;
//     use std::time::Duration;
//
//     define_language! {
//         enum SimpleLanguage {
//             Num(i32),
//             "+" = Add([Id; 2]),
//             "*" = Mul([Id; 2]),
//             Symbol(Symbol),
//         }
//     }
//
//     fn make_rules() -> Vec<Rewrite<SimpleLanguage, ()>> {
//         vec![
//             rewrite!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
//             rewrite!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),
//             rewrite!("add-0"; "(+ ?a 0)" => "?a"),
//             rewrite!("mul-0"; "(* ?a 0)" => "0"),
//             rewrite!("mul-1"; "(* ?a 1)" => "?a"),
//         ]
//     }
//
//     const NODE_LIMIT: usize = 10000;
//     const TIME_LIMIT: usize = 10;
//
//     #[test]
//     fn test_build_and_close() {
//         let mut pool = PoolManager::new(
//             "test",
//             1,
//             1.0,
//             1,
//             true,
//             "(* 0 42)".parse().unwrap(),
//             make_rules(),
//             NODE_LIMIT,
//             TIME_LIMIT,
//         );
//         assert_eq!(pool.has_idle_server(), true);
//         pool.assign_nothing_task();
//         println!("occupancy: {}", pool.occupancy());
//         pool.get_complete_task();
//         println!("after occupancy: {}", pool.occupancy());
//
//         // let a = vec![1, 2, 3];
//         // for (i, j) in a.iter().enumerate() {
//         //     println!("{} - {}", i, j);
//         // }
//         thread::sleep(Duration::from_secs(1));
//
//         pool.close();
//     }
//
//     #[test]
//     fn test_poll_channel() {
//         let worker_num = 5;
//         let mut pool = PoolManager::new(
//             "test",
//             worker_num,
//             1.0,
//             1,
//             true,
//             "(* 0 42)".parse().unwrap(),
//             make_rules(),
//             NODE_LIMIT,
//             TIME_LIMIT,
//         );
//         pool.assign_nothing_task();
//         pool.assign_nothing_task();
//         pool.assign_nothing_task();
//         pool.assign_nothing_task();
//
//         println!("occupancy: {}", pool.occupancy());
//         pool.close();
//         println!("test_pool done");
//     }
// }
