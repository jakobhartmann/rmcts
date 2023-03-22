use egg::*;
use rmcts::*;

define_language! {
    enum SimpleLanguage {
        Num(i32),
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        Symbol(Symbol),
    }
}

fn make_rules() -> Vec<Rewrite<SimpleLanguage, ()>> {
    vec![
        rewrite!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),
        rewrite!("add-0"; "(+ ?a 0)" => "?a"),
        rewrite!("mul-0"; "(* ?a 0)" => "0"),
        rewrite!("mul-1"; "(* ?a 1)" => "?a"),
    ]
}

/// parse an expression, simplify it using egg, and pretty print it back out
fn simplify(s: &str) -> String {
    // parse the expression, the type annotation tells it which Language to use
    let expr: RecExpr<SimpleLanguage> = s.parse().unwrap();

    // simplify the expression using a Runner, which creates an e-graph with
    // the given expression and runs the given rules over it
    let runner = Runner::default().with_expr(&expr);
    let root = runner.roots[0];

    // base cost
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (base_cost, _base) = extractor.find_best(root);

    // best
    let runner = runner.run(&make_rules());
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (best_cost, best) = extractor.find_best(root);

    println!(
        "Simplified {} to {} with base_cost {} -> cost {}",
        expr, best, base_cost, best_cost
    );
    best.to_string()
}

#[test]
fn simple_egg_test() {
    assert_eq!(simplify("(* 0 42)"), "0");
    assert_eq!(simplify("(+ 0 (* 1 foo))"), "foo");
}

#[test]
fn simple_mcts_geb_test() {
    let expr = "(* 0 42)";
    let rws = make_rules();
    run::run_mcts(expr, rws);
}