#![allow(unused)]

mod eg_env;
mod env;
mod node;
mod pool_manager;
mod run;
mod tree;
mod workers;
mod math;
mod prop;
mod utils;

use std::fs::create_dir_all;
use std::path::Path;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use serde_json::{to_string, json};

use egg::*;
use crate::run::MCTSArgs;
use crate::math::*;
use crate::prop::*;
use crate::utils::save_data_to_file;

define_language! {
    enum SimpleLanguage {
        Num(i32),
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        Symbol(Symbol),
    }
}

fn make_rules() -> Vec<egg::Rewrite<SimpleLanguage, ()>> {
    vec![
        rewrite!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),
        rewrite!("add-0"; "(+ ?a 0)" => "?a"),
        rewrite!("mul-0"; "(* ?a 0)" => "0"),
        rewrite!("mul-1"; "(* ?a 1)" => "?a"),
    ]
}

// from egg, and we need Clone trait
#[derive(Debug, Clone, Serialize)]
pub struct AstSize;
impl<L: Language> CostFunction<L> for AstSize {
    type Cost = usize;
    fn cost<C>(&mut self, enode: &L, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        enode.fold(1, |sum, id| sum.saturating_add(costs(id)))
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "lp")))]
impl<L: Language, N: Analysis<L>> LpCostFunction<L, N> for AstSize {
    fn node_cost(&mut self, _egraph: &egg::EGraph<L, N>, _eclass: Id, _enode: &L) -> f64 {
        1.0
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Settings {
    // expression
    domain: String,
    expression_depth: u32,
    expression_seed: u64,
    // experiment tracking
    experiments_base_path: String,
    // mcts
    budget: u32,
    max_sim_step: u32,
    gamma: f32,
    expansion_worker_num: usize,
    simulation_worker_num: usize,
    lp_extract: bool,
    cost_threshold: usize,
    iter_limit: usize,
    prune_actions: bool,
    rollout_strategy: String,
    subtree_caching: bool,
    select_max_uct_action: bool,
    // egg
    node_limit: usize,
    time_limit: usize,
}

fn run_math_experiments (settings: Settings, run_egg: bool) {
    // Create output dir
    let experiment_name = settings.domain.clone() + "_" + &settings.expression_depth.to_string() + "_" + &settings.expression_seed.to_string();
    let output_dir = Path::new(&settings.experiments_base_path).join(experiment_name);
    if output_dir.exists() {
        println!("You are overwriting existing data!");
    } else {
        create_dir_all(&output_dir);
    }

    // Save settings
    save_data_to_file(&settings, &output_dir, "settings.txt");

    // Generate and save expression
    let start_expr = math::build_rand_expr(settings.expression_seed, settings.expression_depth);
    save_data_to_file(&start_expr, &output_dir, "start_expr.txt");

    // ### Start: egg ###
    if run_egg {
        // Create runner
        let runner: Runner<Math, math::ConstantFold> = Runner::default()
            .with_expr(&start_expr)
            .with_iter_limit(settings.iter_limit)
            .with_node_limit(settings.node_limit);
        let root = runner.roots[0];

        // Base cost
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (base_cost, _base_expr) = extractor.find_best(root);

        // Run egg
        let start_time = Instant::now();
        let runner = runner.run(&math::rules());
        let duration = start_time.elapsed();
        runner.print_report();

        // Best cost and expression
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (egg_cost, egg_expr) = extractor.find_best(root);
        println!("Simplified expression {} to {} with base_cost {} -> cost {}", start_expr, egg_expr, base_cost, egg_cost);

        // Save best expression, iteration data and runner report
        save_data_to_file(&egg_expr, &output_dir, "egg_expr.txt");
        save_data_to_file(&runner.iterations, &output_dir, "egg_iteration_data.txt");
        save_data_to_file(&runner.report(), &output_dir, "egg_runner_report.txt");

        // Save stats
        let egg_stats = json!({
            "base_cost": base_cost,
            "best_cost": egg_cost,
            "optimization_time": duration,
        });
        save_data_to_file(&egg_stats, &output_dir, "egg_stats.txt");
    }
    // ### End: egg ###
    

    // ### Start: MCTS ###
    let args = MCTSArgs {
        // mcts
        budget: settings.budget,
        max_sim_step: settings.max_sim_step,
        gamma: settings.gamma,
        expansion_worker_num: settings.expansion_worker_num,
        simulation_worker_num: settings.simulation_worker_num,
        lp_extract: settings.lp_extract,
        cost_threshold: settings.cost_threshold,
        iter_limit: settings.iter_limit,
        prune_actions: settings.prune_actions,
        rollout_strategy: settings.rollout_strategy,
        subtree_caching: settings.subtree_caching,
        select_max_uct_action: settings.select_max_uct_action,
        // experiment tracking
        output_dir: output_dir,
        // egg
        node_limit: settings.node_limit,
        time_limit: settings.time_limit,
    };

    let runner = Runner::default().with_expr(&start_expr);
    let root = runner.roots[0];
    run::run_mcts(runner.egraph, root, math::rules(), AstSize, Some(args));
}

fn run_prop_experiments(settings: Settings, run_egg: bool) {
    // Create output dir
    let experiment_name = settings.domain.clone() + "_" + &settings.expression_depth.to_string() + "_" + &settings.expression_seed.to_string();
    let output_dir = Path::new(&settings.experiments_base_path).join(experiment_name);
    if output_dir.exists() {
        println!("You are overwriting existing data!");
    } else {
        create_dir_all(&output_dir);
    }

    // Save settings
    save_data_to_file(&settings, &output_dir, "settings.txt");

    // Generate and save expression
    let start_expr = prop::build_rand_expr(settings.expression_seed, settings.expression_depth);
    save_data_to_file(&start_expr, &output_dir, "start_expr.txt");

    // ### Start: egg ###
    if run_egg {
        // Create runner
        let runner: Runner<Prop, prop::ConstantFold> = Runner::default()
            .with_expr(&start_expr)
            .with_iter_limit(settings.iter_limit)
            .with_node_limit(settings.node_limit);
        let root = runner.roots[0];

        // Base cost
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (base_cost, _base_expr) = extractor.find_best(root);

        // Run egg
        let start_time = Instant::now();
        let runner = runner.run(&prop::rules());
        let duration = start_time.elapsed();
        runner.print_report();

        // Best cost and expression
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (egg_cost, egg_expr) = extractor.find_best(root);
        println!("Simplified expression {} to {} with base_cost {} -> cost {}", start_expr, egg_expr, base_cost, egg_cost);

        // Save best expression, iteration data and runner report
        save_data_to_file(&egg_expr, &output_dir, "egg_expr.txt");
        save_data_to_file(&runner.iterations, &output_dir, "egg_iteration_data.txt");
        save_data_to_file(&runner.report(), &output_dir, "egg_runner_report.txt");

        // Save stats
        let egg_stats = json!({
            "base_cost": base_cost,
            "best_cost": egg_cost,
            "optimization_time": duration,
        });
        save_data_to_file(&egg_stats, &output_dir, "egg_stats.txt");
    }
    // ### End: egg ###
    

    // ### Start: MCTS ###
    let args = MCTSArgs {
        // mcts
        budget: settings.budget,
        max_sim_step: settings.max_sim_step,
        gamma: settings.gamma,
        expansion_worker_num: settings.expansion_worker_num,
        simulation_worker_num: settings.simulation_worker_num,
        lp_extract: settings.lp_extract,
        cost_threshold: settings.cost_threshold,
        iter_limit: settings.iter_limit,
        prune_actions: settings.prune_actions,
        rollout_strategy: settings.rollout_strategy,
        subtree_caching: settings.subtree_caching,
        select_max_uct_action: settings.select_max_uct_action,
        // experiment tracking
        output_dir: output_dir,
        // egg
        node_limit: settings.node_limit,
        time_limit: settings.time_limit,
    };

    let runner = Runner::default().with_expr(&start_expr);
    let root = runner.roots[0];
    run::run_mcts(runner.egraph, root, prop::rules(), AstSize, Some(args));
}

fn main() {
    let mut settings = Settings {
        // expression
        domain: String::from("prop"),
        expression_depth: 10,
        expression_seed: 0,
        // experiment tracking
        experiments_base_path: String::from("/usr/experiments/prop/"),
        // mcts
        budget: 512,
        max_sim_step: 10,
        gamma: 0.99,
        expansion_worker_num: 1,
        simulation_worker_num: 1,
        lp_extract: false,
        cost_threshold: 1,
        iter_limit: 100,
        prune_actions: false,
        rollout_strategy: String::from("random"),
        subtree_caching: false,
        select_max_uct_action: true,
        // egg
        node_limit: 5_000,
        time_limit: 1,
    };

    // Experiment 1: egg vs. rmcts
    let expressions_seeds = vec![0, 1, 2, 3, 4];
    let node_limits = vec![5_000, 10_000];
    let mut settings1 = settings.clone();

    for seed in &expressions_seeds {    
        for node_limit in &node_limits {
            settings1.expression_seed = *seed;
            settings1.node_limit = *node_limit;
            settings1.experiments_base_path = String::from("/usr/experiments/prop/egg_vs_rmcts/".to_owned() + &node_limit.to_string());
            run_prop_experiments(settings1.clone(), true);
        }
    }

    // Experiment 2: subtree caching and action selection strategy
    let subtree_caching_options = vec![false, true];
    let action_select_strategies = vec![false, true];
    let expressions_seeds = vec![0, 1, 2, 3, 4];
    let mut settings2 = settings.clone();

    for seed in &expressions_seeds {
        for option in &subtree_caching_options {
            for strategy in &action_select_strategies {
                settings2.expression_seed = *seed;
                settings2.subtree_caching = *option;
                settings2.select_max_uct_action = *strategy;
                settings2.experiments_base_path = String::from("/usr/experiments/prop/subtree_caching/".to_owned() + &option.to_string() + "/" + &strategy.to_string());
                run_prop_experiments(settings2.clone(), true);
            }
        }
    }

    // Experiment 3: rmcts with and without action pruning
    let expressions_seeds = vec![0, 1, 2, 3, 4];
    let experiments = vec![(false, String::from("random")), (true, String::from("pruning")), (true, String::from("heavy"))];
    let mut settings3 = settings.clone();

    for experiment in &experiments {
        for seed in &expressions_seeds {
            let (prune_actions, rollout_strategy) = experiment;
            settings3.expression_seed = *seed;
            settings3.prune_actions = *prune_actions;
            settings3.rollout_strategy = rollout_strategy.clone();
            settings3.experiments_base_path = String::from("/usr/experiments/prop/pruning/".to_owned() + &prune_actions.to_string() + "_" + &rollout_strategy.to_owned());
            run_prop_experiments(settings3.clone(), true);
        }
    }

    // Experiment 4: rmcts with different budgets
    let expressions_seeds = vec![0, 1, 2, 3, 4];
    let budgets = vec![64, 128, 256, 512, 1024];
    let mut settings4 = settings.clone();

    for budget in &budgets {
        for seed in &expressions_seeds {
            settings4.expression_seed = *seed;
            settings4.budget = *budget;
            settings4.experiments_base_path = String::from("/usr/experiments/prop/budget/".to_owned() + &budget.to_string());
            run_prop_experiments(settings4.clone(), true);
        }
    }

    // Experiment 5: rmcts with different simulation depths
    let expressions_seeds = vec![0, 1, 2, 3, 4];
    let max_sim_steps = vec![5, 10, 15];
    let mut settings5 = settings.clone();

    for max_sim_step in &max_sim_steps {
        for seed in &expressions_seeds {
            settings5.expression_seed = *seed;
            settings5.max_sim_step = *max_sim_step;
            settings5.experiments_base_path = String::from("/usr/experiments/prop/simulation_depth/".to_owned() + &max_sim_step.to_string());
            run_prop_experiments(settings5.clone(), true);
        }
    }

    // let args = MCTSArgs {
    //     // mcts
    //     budget: 12,
    //     max_sim_step: 10,
    //     gamma: 0.99,
    //     expansion_worker_num: 1,
    //     simulation_worker_num: 1,
    //     lp_extract: false,
    //     cost_threshold: 1,
    //     iter_limit: 100,
    //     prune_actions: true,
    //     rollout_strategy: String::from("heavy"), // random, pruning, heavy (if pruning is chosen, prune actions has to be set to true, otherwise it has no effect)
    //     subtree_caching: true,
    //     select_max_uct_action: false,
    //     // experiment tracking
    //     output_dir: Path::new("/usr/experiments/tests/").to_path_buf(),
    //     // egg
    //     node_limit: 10_000,
    //     time_limit: 1,
    // };

    // let expr = "(* (* 0 42) 1)".parse().unwrap();
    // let runner = Runner::default().with_expr(&expr);
    // let root = runner.roots[0];
    // run::run_mcts(runner.egraph, root, make_rules(), AstSize, Some(args));
}
