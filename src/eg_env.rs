use std::time::Duration;
use egg::{
    Analysis, CostFunction, EGraph, Extractor, Id, Language, Rewrite,
    Runner, SimpleScheduler, StopReason,
};
use crate::env::Info;

#[derive(Clone)]
pub struct Ckpt<L, N>
where
    L: Language + 'static + egg::FromOp + std::marker::Send + std::fmt::Display,
    N: Analysis<L> + Clone + 'static + std::default::Default + std::marker::Send,
    N::Data: Clone,
    <N as Analysis<L>>::Data: Send,
{
    pub cnt: u32,
    pub sat_counter: usize,
    pub egraph: EGraph<L, N>,
    pub root_id: Id,
    pub last_cost: usize,
}

pub struct EgraphEnv<L, N, CF>
where
    L: Language + 'static + egg::FromOp + std::marker::Send + std::fmt::Display,
    N: Analysis<L> + Clone + 'static + std::default::Default + std::marker::Send,
    N::Data: Clone,
    <N as Analysis<L>>::Data: Send,
    CF: CostFunction<L> + Clone + std::marker::Send + 'static,
    usize: From<<CF as CostFunction<L>>::Cost>,
{
    init_egraph: EGraph<L, N>,
    pub egraph: EGraph<L, N>,
    cf: CF,
    #[allow(dead_code)]
    lp_extract: bool,
    pub root_id: Id,
    num_rules: usize,
    rules: Vec<Rewrite<L, N>>,

    prune_actions: bool,

    node_limit: usize,
    time_limit: std::time::Duration,

    pub base_cost: usize,
    pub last_cost: usize,
    cnt: u32,
    sat_counter: usize,
}

impl<L, N, CF> EgraphEnv<L, N, CF>
where
    L: Language + 'static + egg::FromOp + std::marker::Send + std::fmt::Display,
    N: Analysis<L> + Clone + 'static + std::default::Default + std::marker::Send,
    N::Data: Clone,
    <N as Analysis<L>>::Data: Send,
    CF: CostFunction<L> + Clone + std::marker::Send + 'static,
    usize: From<<CF as CostFunction<L>>::Cost>,
{
    pub fn new(
        egraph: EGraph<L, N>,
        root_id: Id,
        rules: Vec<Rewrite<L, N>>,
        cf: CF,
        lp_extract: bool,
        prune_actions: bool,
        node_limit: usize,
        time_limit: usize,
    ) -> Self {
        assert!(!lp_extract); // TODO lp_extract only gets expr, but we need to compute cost
                              // init expr cost; NOTE: expr length != expr cost
        let (base_cost, _) = Extractor::new(&egraph, cf.clone()).find_best(root_id);
        let base_cost = usize::try_from(base_cost).unwrap();
        // let len = expr.as_ref().len();
        // assert_eq!(len, base_cost);
        //
        EgraphEnv {
            init_egraph: egraph,
            egraph: EGraph::default(),
            cf: cf,
            lp_extract: lp_extract,
            root_id: root_id,
            num_rules: rules.len(),
            rules: rules,
            prune_actions: prune_actions,
            node_limit: node_limit,
            time_limit: Duration::from_secs(time_limit.try_into().unwrap()),

            base_cost: base_cost,
            last_cost: 0,
            cnt: 0,
            sat_counter: 0,
        }
    }

    pub fn reset(&mut self) {
        self.cnt = 0;
        self.sat_counter = 0;
        self.egraph = self.init_egraph.clone();
        self.last_cost = self.base_cost;
    }

    pub fn step(&mut self, action: usize) -> ((), f32, bool, Info) {
        // run egg
        let egraph = std::mem::take(&mut self.egraph);
        let rule = vec![self.rules[action].clone()];
        let runner: Runner<L, N> = Runner::default()
            .with_egraph(egraph)
            .with_iter_limit(1)
            .with_node_limit(self.node_limit)
            .with_time_limit(self.time_limit)
            .with_scheduler(SimpleScheduler)
            .run(&rule);
        let report = runner.report();

        // reclaim the partial egraph
        self.egraph = runner.egraph;

        // let num_applications: usize = runner
        //     .iterations
        //     .iter()
        //     .map(|i| i.applied.values().sum::<usize>())
        //     .sum();

        // run extract
        // let (cost, _) = Extractor::new(&self.egraph, self.cf.clone()).find_best(self.root_id);
        let cost = Extractor::new(&self.egraph, self.cf.clone()).find_best_cost(self.root_id);
        let best_cost = usize::try_from(cost).unwrap();

        // compute transition
        self.cnt += 1;
        let mut done = false;
        match runner.stop_reason.as_ref().unwrap() {
            StopReason::NodeLimit(_) => {
                done = true;
                self.sat_counter = 0;
                // println!(
                //     "EGG NodeLimit {}s - {}s - {} - {} - {}",
                //     node_limit,
                //     report.total_time,
                //     report.egraph_nodes,
                //     report.egraph_classes,
                //     report.memo_size,
                // );
            }
            StopReason::TimeLimit(time) => {
                // TODO this indicates egraph is exploded?
                done = true;
                println!(
                    "EGG TimeLimit {}s - {}s - {} - {} - {}",
                    time,
                    report.total_time,
                    report.egraph_nodes,
                    report.egraph_classes,
                    report.memo_size,
                );
            }
            StopReason::Saturated => {
                // TODO sat_counter is enough to indicate saturation?
                self.sat_counter += 1;
                if self.sat_counter == (self.num_rules) {
                    done = true;
                }
            }
            StopReason::IterationLimit(_) => self.sat_counter = 0,
            _ => self.sat_counter = 0,
        }
        // compute reward
        let reward = std::cmp::max(self.last_cost - best_cost, 0);
        self.last_cost = best_cost;
        let info = Info {
            report: report,
            best_cost: best_cost,
        };

        ((), (reward as f32), done, info)
    }

    // immediately extract and get reward
    // pub fn get_reward(&self) -> f32 {
    //     let extractor = Extractor::new(&self.egraph, AstSize);
    //     let (best_cost, _) = extractor.find_best(self.root_id);
    //     let reward = std::cmp::max(self.last_cost - best_cost, 0);

    //     reward as f32
    // }

    pub fn get_action_space(&self) -> usize {
        self.num_rules
    }

    pub fn checkpoint(&self) -> Ckpt<L, N> {
        Ckpt {
            cnt: self.cnt,
            sat_counter: self.sat_counter,
            egraph: self.egraph.clone(),
            root_id: self.root_id.clone(),
            last_cost: self.last_cost,
        }
    }

    pub fn restore(&mut self, checkpoint_data: Ckpt<L, N>) {
        self.cnt = checkpoint_data.cnt;
        self.sat_counter = checkpoint_data.sat_counter;
        self.egraph = checkpoint_data.egraph;
        self.root_id = checkpoint_data.root_id;
        self.last_cost = checkpoint_data.last_cost;
    }

    pub fn action_pruning(&mut self, mut children_saturated: Vec<bool>) -> (Vec<bool>, usize) {
        if self.prune_actions {
            if children_saturated.iter().filter(|x| **x).count() > 0 {
                panic!("At least one child is already saturated. Should this ever happen?");
            }

            // Create runner with egraph
            let egraph = std::mem::take(&mut self.egraph);
            let mut runner = Runner::default().with_egraph(egraph);

            // If the runner is not clean, rebuild the egraph
            if !runner.egraph.clean {
                runner.egraph.rebuild();
            }

            // Iterate over all single-pattern rewrite rules and check if the source pattern is found at least once in the egraph. IMPORTANT: Search with limit, otherwise we get an OOM for larger egraphs!
            // If the source pattern is not found, mark the child accordingly and increase the saturation counter of the environment
            for (i, rewrite) in self.rules.iter().enumerate() {
                if rewrite.searcher.search_with_limit(&runner.egraph, 1).len() == 0 {
                    if !children_saturated[i] {
                        children_saturated[i] = true;
                        self.sat_counter += 1;
                    }
                }
            }

            // Reclaim the partial graph
            self.egraph = runner.egraph;
        }

        // Calculate the number of saturated children and return it together with the corresponding list
        let children_saturated_cnt = children_saturated.iter().filter(|x| **x).count();
        (children_saturated, children_saturated_cnt)
    }
}
