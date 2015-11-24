#!/usr/bin/env python

# Future imports
from __future__ import absolute_import
from __future__ import print_function

# Python modules
import argparse
import imp
import logging
import os
import subprocess
import sys

# LLDB modules
import use_lldb_suite

def generate(options):
    include_folder = os.path.join(options.src_root, "include")
    in_file = os.path.join(options.src_root, "scripts", "lldb.swig")
    include_folder = os.path.normcase(include_folder)

    for lang in options.languages:
        lang = lang.lower()
        out_dir = os.path.join(options.target_dir, lang.title())
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = os.path.join(out_dir, "LLDBWrap{}".format(lang.title()))
        swig_command = [
            options.swig_executable,
            "-c++",
        ]
        swig_command.append("-" + lang)
        if lang == "python":
            swig_command.append("-threads")

        swig_command.extend([
            "-I" + include_folder,
            "-D__STDC_LIMIT_MACROS",
            "-D__STDC_CONSTANT_MACROS",
            "-outdir", out_dir,
            "-o", out_file,
            in_file
        ])

        logging.info("generating swig {} bindings into {}".format(lang, out_dir))
        logging.debug("swig command line: {}".format(swig_command))
        try:
            # Execute swig
            swig_output = subprocess.check_output(
                swig_command, stderr=subprocess.STDOUT, universal_newlines=True)

            logging.info("swig generation succeeded")
            if swig_output is not None and len(swig_output) > 0:
                logging.info("swig output: %s", swig_output)
        except subprocess.CalledProcessError as e:
            logging.error("An error occurred executing swig.  returncode={}".format(e.returncode))
            logging.error(e.output)
