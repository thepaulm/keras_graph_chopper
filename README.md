# Keras Graph Chopper

Python script to chop apart existing keras models and gen a new model with arbitrary inputs and
outputs.

Sometimes you have a big keras graph model and you want to chop it apart into a subgraph with
new inputs and outputs. This script does that for you.

Some example usage:

```
./keras_graph_chopper.py --source-model=~/models/original_model.h5 \
			--dest-model=~/models/new_model.h5 \
			--input-names=post_precessing,first_input,injected_feature \
			--output-names=intermiate_layer,original_output
```

This script will generate new inputs, resolve all layer dependencies, build new subgraph, and
save it.

Issue reports and PRs welcome.
