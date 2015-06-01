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

omp_root?=.
include $(omp_root)/tools/common.inc
.PHONY: default all omp

default: omp

all: omp stubs

omp: info mkdir
	@echo $(omp_root)/tools/build.pl $(build_args) --arch=$(arch) --mode=$(mode) lib inc common -- -j$(jobs)
	$(omp_root)/tools/build.pl $(build_args) --arch=$(arch) --mode=$(mode) lib inc common -- -j$(jobs)

stubs: mkdir
	@echo $(omp_root)/tools/build.pl $(build_args) --arch=$(arch) --mode=$(mode) --stubs lib inc common -- -j$(jobs)
	$(omp_root)/tools/build.pl $(build_args) --arch=$(arch) --mode=$(mode) --stubs lib inc common -- -j$(jobs)

.PHONY: clean info

clean:
	$(shell $(RM) -rf $(omp_root)$(SLASH)tmp)
	@echo clean done

mkdir:
	$(shell $(MD) $(omp_root)$(SLASH)tmp >$(NUL) 2>$(NUL))
	@echo Created $(omp_root)$(SLASH)tmp directory

info:
	@echo omp_root=$(omp_root)
	@echo omp_os=$(omp_os)
	@echo arch=$(arch)
ifeq "$(arch)" "mic"
	@echo mic_arch=$(mic_arch)	
endif
	@echo compiler=$(compiler)	
	@echo mic=$(mic)
	@echo mode=$(mode)
	@echo jobs=$(jobs)



libomp_path=$(shell $(omp_root)/tools/check-openmp.pl)
test_path=$(shell $(omp_root)/tools/check-openmp-test.pl)

# Please do not change this rule.
# -------------------------------
# Intentionally changing directory into "testsuite" in order to generate output results and errors
# over there and keep this directory clean.
test: omp	
	@$(Verb) if which llvm-lit &> /dev/null; then \
        if [ -d "$(omp_root)$(SLASH)..$(SLASH)testsuite$(SLASH)LLVM-IR" ] ; then \
			export TESTSUITE_TEMP=$(realpath $(omp_root))$(SLASH)tmp ; \
			export LIBRARY_PATH=$(libomp_path):$(LIBRARY_PATH) ; \
			export C_INCLUDE_PATH=$(libomp_path)$(SLASH)..$(SLASH)..$(SLASH)common$(SLASH)include:$(C_INCLUDE_PATH) ; \
			export LD_LIBRARY_PATH=$(libomp_path):$(LD_LIBRARY_PATH) ; \
			export DYLD_LIBRARY_PATH=$(libomp_path):$(DYLD_LIBRARY_PATH) ; \
			cd $(omp_root)$(SLASH)..$(SLASH)testsuite ; \
			make ctest ; \
			python adding_xfails.py ; \
			llvm-lit -j 1 $(realpath $(omp_root))$(SLASH)..$(SLASH)testsuite$(SLASH)LLVM-IR$(SLASH)$(test_path) -v ; \
        else \
        	echo "No test directory" ; exit 1; \
        fi; else echo "No llvm-lit in $(PATH)"; exit 1; fi

test-clean:
	make -C ..$(SLASH)testsuite cleanall


