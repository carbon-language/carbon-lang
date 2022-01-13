# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from itertools import chain

def in_source_file(source_files, step_info):
    if not step_info.current_frame:
        return False
    if not step_info.current_location.path:
        return False
    if not os.path.exists(step_info.current_location.path):
        return False
    return any(os.path.samefile(step_info.current_location.path, f) \
               for f in source_files)

def update_step_watches(step_info, watches, commands):
    watch_cmds = ['DexUnreachable', 'DexExpectStepOrder']
    towatch = chain.from_iterable(commands[x]
                                  for x in watch_cmds
                                  if x in commands)
    try:
        # Iterate over all watches of the types named in watch_cmds
        for watch in towatch:
            loc = step_info.current_location
            if (loc.path != None
                    and os.path.exists(loc.path)
                    and os.path.samefile(watch.path, loc.path)
                    and watch.lineno == loc.lineno):
                result = watch.eval(step_info)
                step_info.watches.update(result)
                break
    except KeyError:
        pass
