#!/usr/bin/env python

# Future imports
from __future__ import absolute_import
from __future__ import print_function

# Python modules
import argparse
import logging
import os
import sys

# LLDB modules
import use_lldb_suite
from lldbsuite.support import fs

def process_args(args):
    """Returns options processed from the provided command line.

    @param args the command line to process.
    """

    class FindLocalSwigAction(argparse.Action):
        def __init__(self, option_strings, dest, **kwargs):
            super(FindLocalSwigAction, self).__init__(option_strings, dest, nargs='?', **kwargs)
        def __call__(self, parser, namespace, values, option_string=None):
            swig_exe = None
            if values is None:
                swig_exe = fs.find_executable('swig')
            else:
                swig_exe = values
            setattr(namespace, self.dest, os.path.normpath(swig_exe))

    # Setup the parser arguments that are accepted.
    parser = argparse.ArgumentParser(
        description='Generate SWIG bindings.')

    # Arguments to control logging verbosity.
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Increase logging verbosity level.")

    parser.add_argument(
        "--local",
        action=FindLocalSwigAction,
        dest="swig_executable",
        help=(
            "Run the copy of swig at the specified location, or search PATH"
            "if the location is omitted"))

    parser.add_argument(
        "--remote",
        action="store",
        help=(
            "Use the given connection string to connect to a remote "
            "generation service"))

    parser.add_argument(
        "--src-root",
        required=True,
        help="The root folder of the LLDB source tree.")

    parser.add_argument(
        "--target-dir",
        default=os.getcwd(),
        help=(
            "Specifies the build dir where the language binding "
            "should be placed"))

    parser.add_argument(
        "--language",
        dest="languages",
        action="append",
        help="Specifies the language to generate bindings for")

    # Process args.
    options = parser.parse_args(args)

    if options.languages is None:
        options.languages = ['python']

    if options.remote is None and options.swig_executable is None:
        logging.error("Must specify either --local or --remote")
        sys.exit(-3)

    # Set logging level based on verbosity count.
    if options.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.NOTSET
    logging.basicConfig(level=log_level)
    logging.info("logging is using level: %d", log_level)

    return options


def run(args):
    options = process_args(args)

    if options.remote is None:
        if not os.path.isfile(options.swig_executable):
            logging.error("Swig executable '%s' does not exist." % options.swig_executable)
        from . import local
        local.generate(options)
    else:
        logging.error("Remote path is not yet implemented!")
