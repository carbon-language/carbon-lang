# common-checks.mk #

#
#//===----------------------------------------------------------------------===//
#//
#//                     The LLVM Compiler Infrastructure
#//
#// This file is dual licensed under the MIT and the University of Illinois Open
#// Source Licenses. See LICENSE.txt for details.
#//
#//===----------------------------------------------------------------------===//
#

# --------------------------------------------------------------------------------------------------
# This file contains really common definitions used by multiple makefiles. Modify it carefully!
# --------------------------------------------------------------------------------------------------

#
# Check tools versions.
#
ifeq "$(clean)" ""    # Do not check tools if clean goal specified.

    check_tools_flags = --make

    # determine if fortran check is required from goals
    # MAKECMDGOALS is like argv for gnu make
    ifneq "$(filter mod all,$(MAKECMDGOALS))" ""
        check_tools_flags += --fortran
    else
        ifeq "$(MAKECMDGOALS)" "" # will default to all if no goals specified on command line
            check_tools_flags += --fortran
        endif
    endif
    ifneq "$(filter gcc clang,$(c))" "" # if build compiler is gcc or clang
        check_tools_flags += --nointel
    endif
    ifeq "$(c)" "clang"
        check_tools_flags += --clang
    endif

    curr_tools := $(strip $(shell $(perl) $(tools_dir)check-tools.pl $(oa-opts) $(check_tools_flags)))

    ifeq "$(curr_tools)" ""
        $(error check-tools.pl failed)
    endif
    ifneq "$(findstring N/A,$(curr_tools))" ""
        missed_tools := $(filter %---_N/A_---,$(curr_tools))
        missed_tools := $(subst =---_N/A_---,,$(missed_tools))
        missed_tools := $(subst $(space),$(comma)$(space),$(missed_tools))
        $(error Development tools not found: $(missed_tools))
    endif
    prev_tools := $(strip $(shell [ -e tools.cfg ] && cat tools.cfg))
    $(call say,Tools  : $(curr_tools))
    ifeq "$(prev_tools)" ""
        # No saved config file, let us create it.
        dummy := $(shell echo "$(curr_tools)" > tools.cfg)
    else
        # Check the saved config file matches current configuration.
        ifneq "$(curr_tools)" "$(prev_tools)"
            # Show the differtence between previous and current tools.
            $(call say,Old tools : $(filter-out $(curr_tools),$(prev_tools)))
            $(call say,New tools : $(filter-out $(prev_tools),$(curr_tools)))
            # And initiate rebuild.
            $(call say,Tools changed$(comma) rebuilding...)
            dummy := $(shell $(rm) .rebuild && echo "$(curr_tools)" > tools.cfg)
        endif
    endif
endif

# Check config.
ifeq "$(curr_config)" ""
    $(error makefile must define `curr_config' variable)
endif
prev_config := $(shell [ -e build.cfg ] && cat build.cfg)
curr_config := $(strip $(curr_config))
ifeq "$(clean)" ""    # Do not check config if clean goal specified.
    $(call say,Config : $(curr_config))
    ifeq "$(prev_config)" ""
        # No saved config file, let us create it.
        dummy := $(shell echo "$(curr_config)" > build.cfg)
    else
        # Check saved config file matches current configuration.
        ifneq "$(curr_config)" "$(prev_config)"
            # Show the differtence between previous and current configurations.
            $(call say,Old config : $(filter-out $(curr_config),$(prev_config)))
            $(call say,New config : $(filter-out $(prev_config),$(curr_config)))
            # And initiate rebuild.
            $(call say,Configuration changed$(comma) rebuilding...)
            dummy := $(shell $(rm) .rebuild && echo "$(curr_config)" > build.cfg)
        endif
    endif
endif

# end of file #

