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
	@echo $(omp_root)/tools/build.pl $(build_args) --arch=$(arch) --mode=$(mode) lib inc common
	$(omp_root)/tools/build.pl $(build_args) --arch=$(arch) --mode=$(mode) lib inc common

stubs: mkdir
	@echo $(omp_root)/tools/build.pl $(build_args) --arch=$(arch) --mode=$(mode) --stubs lib inc common 
	$(omp_root)/tools/build.pl $(build_args) --arch=$(arch) --mode=$(mode) --stubs lib inc common

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
	@echo compiler=$(compiler)	
	@echo mic=$(mic)
	@echo mode=$(mode)

