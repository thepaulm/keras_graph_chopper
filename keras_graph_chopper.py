#!/usr/bin/env python

from __future__ import print_function

import argparse
from keras.layers import Input
from keras.models import Model
from keras.models import load_model
from keras.engine.topology import InputLayer

verbose = False


def inspect(m):
    return [(i, l, l.name, l.input_shape, l.output_shape) for i, l in enumerate(m.layers)]


def get_layer_name(m, name):
    return [l for l in m.layers if l.name == name]


def outbound_layer(l):
    return [n.outbound_layer for n in l.outbound_nodes]


def inbound_layers(l):
    return [n.inbound_layers for n in l.inbound_nodes]


def input_to(l):
    input_shape = map(int, l.get_input_at(0).shape[1:])
    return Input(shape=input_shape, name='input_' + l.name)


def copy_layer(l):
    newl = l.from_config(l.get_config())
    return newl


def fragment_copy(start, n):
    global verbose

    outs = [n]
    t = start
    end = None

    expected_ins = 1
    if isinstance(t, list):
        expected_ins = len(t)

    while len(outs) == 1 and \
            len(inbound_layers(outs[0])) == 1 and \
            len(inbound_layers(outs[0])[0]) == expected_ins:

        expected_ins = 1

        n = outs[0]
        if verbose:
            print("copy layer: ", n.name)
        end = copy_layer(n)
        t = end(t)
        try:
            end.set_weights(n.get_weights())
        except Exception as e:
            print(e)
            return start, end
        outs = outbound_layer(n)

    return start, end, n


class Fragment(object):
    def __init__(self, begin, end, orig_end):
        self.begin = begin
        self.end = end
        self.orig_end = orig_end

    def __str__(self):
        return "[begin: {}, end: {}, orig: {}]".format(self.begin, self.end, self.orig_end)


def model_chopper(m, input_names, output_names):
    global verbose

    inputs = [m.get_layer(name=name) for name in input_names]
    outputs = [m.get_layer(name=name) for name in output_names]

    new_inputs = []
    new_outputs = []

    frags = []
    # Go through all inputs, gen them up, and copy as far as we can
    for input in inputs:
        # If one of these is already InputLayer, then move one up and things
        # will work out.
        if isinstance(input, InputLayer):
            input = outbound_layers(input)[0]
        new_input = input_to(input)
        begin, end, orig = fragment_copy(new_input, input)
        frags.append(Fragment(begin, end, orig))
        new_inputs.append(new_input)

    # Now we have graph fragments
    while len(frags) != 0:
        clean_frags = []
        new_frags = []

        # Go through each gragment
        for f in frags:

            if f in clean_frags:
                continue

            if verbose:
                print("Fragment: ", f)

            if f.orig_end in outputs:
                # If this fragment ends in an output then it is complete
                new_outputs.append(f.end)
                clean_frags.append(f)
            else:
                # XXX bug - if the output is also a dep we won't find the dep

                # go through each unmet dep
                n = outbound_layer(f.orig_end)
                assert(len(n) == 1)
                n = n[0]  # n is now the next uncopied node

                dep_list = inbound_layers(n)
                assert(len(dep_list) == 1)
                dep_list = dep_list[0]

                # make sure we have answers for each of the deps in the dep_list
                feed_list = []
                for dep in dep_list:
                    ok = False
                    for fdep in frags:
                        if fdep.orig_end == dep:
                            ok = True
                            # fragment will be clean because we're about to feed
                            clean_frags.append(fdep)
                            feed_list.append(fdep.end)
                    assert(ok)

                feed_list = [fl.output for fl in feed_list]
                start, end, orig = fragment_copy(feed_list, n)
                new_frags.append(Fragment(start, end, orig))

        for f in new_frags:
            frags.append(f)

        for f in clean_frags:
            if f in frags:
                frags.remove(f)

    new_outputs = [l.output for l in new_outputs]
    return Model(inputs=new_inputs, outputs=new_outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-model', help="Source model h5 file", required=True)
    parser.add_argument('--dest-model', help="Dest model h5 file", required=True)
    parser.add_argument('--input-names',
                        help="Comma separated list of input layer names", required=True)
    parser.add_argument('--output-names',
                        help="Comma separated list of output layer names", required=True)
    parser.add_argument('--verbose', help="Print every copied layer", default=False,
                        action='store_true')

    args = parser.parse_args()
    global verbose
    verbose = args.verbose

    m = load_model(args.source_model)
    sub = model_chopper(m,
                        input_names=args.input_names.split(','),
                        output_names=args.output_names.split(','))
    sub.save(args.dest_model, include_optimizer=False)
    print("Saved to ", args.dest_model)
    print("Inputs: ")
    print(sub.inputs)
    print("Outputs: ")
    print(sub.outputs)

if __name__ == '__main__':
    main()
