from omega import automata


if __name__ == '__main__':
    g = automata.TransitionSystem()
    g.owner = 'env'
    g.vars = dict(x='bool', y=(0, 5))
    g.env_vars.add('x')

    g.add_nodes_from(range(5))
    g.add_edge(0, 1, formula="x /\ (y' = 4)")
    g.add_edge(0, 0, formula=" ~ x")
    g.add_edge(1, 2, formula="(y' = 3)")
    g.add_edge(2, 3, formula="(y' = y - 1)")
    g.add_edge(3, 4, formula="y' < y")
    g.add_edge(4, 0, formula="y' = 5")
    g.initial_nodes.add(0)