#
##//===----------------------------------------------------------------------===//
#//
#//                     The LLVM Compiler Infrastructure
#//
#// This file is dual licensed under the MIT and the University of Illinois Open
#// Source Licenses. See LICENSE.txt for details.
#//
#//===----------------------------------------------------------------------===//
#

# MAKEFILE PARAMETERS
#
# root_dir - path to root directory of liboffload
# build_dir - path to build directory
# mpss_dir - path to root directory of mpss
# mpss_version - version of the mpss (e.g., version "3.3.x" would be "33")
# libiomp_host_dir - path to host libiomp directory (unnecessary if compiler_host is icc)
# libiomp_target_dir - path to target libiomp directory (unnecesarry if compiler_target is icc)
# omp_header_dir - path to omp.h (unnecessary if compiler_host and compiler_target are icc)
# os_host - host operating system
# os_target - target operating system
# compiler_host - host compiler
# compiler_target - target compiler
# options_host - additional options for host compiler
# options_target - additional options for target compiler
#

# Directories
root_dir?=.
build_dir?=$(root_dir)/build
build_host_dir=$(build_dir)/host
build_target_dir=$(build_dir)/target
obj_host_dir=$(build_dir)/obj_host
obj_target_dir=$(build_dir)/obj_target
source_dir=$(root_dir)/src
imported_dir=$(source_dir)/imported

# OS
os_host?=linux
os_target?=linux
ifneq ($(os_host)_$(os_target), linux_linux)
  $(error "Only linux is supported")
endif

# Compilers
compiler_host?=gcc
compiler_target?=gcc

# MPSS
mpss_version?=30
mpss_dir?=/
mpss_present=$(shell if test -d $(mpss_dir); then echo OK; else echo KO; fi)
ifneq ($(mpss_present), OK)
  $(error "Cannot find MPSS directory $(mpss_dir)")
endif

ifeq ($(shell test $(mpss_version) -gt 33; echo $$?), 0)
  coi_dir=$(mpss_dir)/sysroots/k1om-mpss-linux/usr
  coi_include=$(coi_dir)/include/intel-coi
  coi_lib_host=$(mpss_dir)/lib64
  coi_lib_device=$(coi_dir)/lib64
else
  coi_dir=$(mpss_dir)/opt/intel/mic/coi
  coi_include=$(coi_dir)/include
  coi_lib_host=$(coi_dir)/host-linux-release/lib
  coi_lib_device=$(coi_dir)/device-linux-release/lib
endif
myo_dir=$(mpss_dir)/opt/intel/mic/myo

# Sources
src_liboffload_common=dv_util.cpp liboffload_error.c liboffload_msg.c offload_common.cpp offload_table.cpp offload_trace.cpp offload_util.cpp

src_liboffload_host=$(src_liboffload_common) cean_util.cpp coi/coi_client.cpp compiler_if_host.cpp offload_engine.cpp offload_env.cpp offload_host.cpp offload_omp_host.cpp offload_timer_host.cpp offload_orsl.cpp orsl-lite/lib/orsl-lite.c offload_myo_host.cpp
src_liboffload_host:=$(foreach file,$(src_liboffload_host),$(source_dir)/$(file))

src_liboffload_target=$(src_liboffload_common) coi/coi_server.cpp compiler_if_target.cpp offload_omp_target.cpp offload_target.cpp offload_timer_target.cpp offload_myo_target.cpp
src_liboffload_target:=$(foreach file,$(src_liboffload_target),$(source_dir)/$(file))

src_ofld=ofldbegin.cpp ofldend.cpp
src_ofld:=$(foreach file,$(src_ofld),$(source_dir)/$(file))

headers=$(wildcard $(source_dir)/*.h) $(wildcard $(source_dir)/coi/*.h) $(wildcard $(source_dir)/orsl-lite/include/*.h)
ifneq ($(omp_header_dir), )
  headers+=$(imported_dir)/omp.h
endif

# Objects
obj_liboffload_host=$(notdir $(src_liboffload_host))
obj_liboffload_host:=$(obj_liboffload_host:.cpp=.o)
obj_liboffload_host:=$(obj_liboffload_host:.c=.o)
obj_liboffload_host:=$(foreach file,$(obj_liboffload_host),$(obj_host_dir)/$(file))

obj_liboffload_target=$(notdir $(src_liboffload_target))
obj_liboffload_target:=$(obj_liboffload_target:.cpp=.o)
obj_liboffload_target:=$(obj_liboffload_target:.c=.o)
obj_liboffload_target:=$(foreach file,$(obj_liboffload_target),$(obj_target_dir)/$(file))

obj_ofld=$(notdir $(src_ofld))
obj_ofld:=$(obj_ofld:.cpp=.o)
obj_ofld_host=$(foreach file,$(obj_ofld),$(build_host_dir)/$(file))
obj_ofld_target=$(foreach file,$(obj_ofld),$(build_target_dir)/$(file))

# Options
opts_common=-O2 -w -fpic -c -DCOI_LIBRARY_VERSION=2 -DMYO_SUPPORT -DOFFLOAD_DEBUG=1 -DSEP_SUPPORT -DTIMING_SUPPORT -I$(coi_include) -I$(myo_dir)/include -I$(source_dir)
ifneq ($(omp_header_dir), )
  opts_common+=-I$(imported_dir)
endif

opts_liboffload=-shared -Wl,-soname,liboffload.so.5 -ldl -lstdc++ -liomp5

opts_liboffload_host=$(opts_liboffload) -L$(coi_lib_host) -lcoi_host -L$(myo_dir)/lib -lmyo-client
ifneq ($(libiomp_host_dir), )
  opts_liboffload_host+=-L$(libiomp_host_dir)
endif

opts_liboffload_target=$(opts_liboffload) -L$(coi_lib_device) -lcoi_device -L$(myo_dir)/lib -lmyo-service
ifneq ($(libiomp_target_dir), )
  opts_liboffload_target+=-L$(libiomp_target_dir)
endif

options_host?=
opts_host=$(options_host) -DHOST_LIBRARY=1 -DMPSS_VERSION=$(mpss_version)
ifeq ($(os_host), linux)
  opts_host+=-DLINUX
endif

options_target?=
opts_target=$(options_target) -DHOST_LIBRARY=0
ifeq ($(os_target), linux)
  opts_target+=-DLINUX
endif
ifeq ($(compiler_target), icc)
  opts_target+=-mmic
endif

# Make targets
.PHONY: all clean info

all: info $(build_host_dir)/liboffload.so $(build_target_dir)/liboffload.so $(obj_ofld_host) $(obj_ofld_target)


$(build_host_dir)/liboffload.so: $(build_host_dir)/liboffload.so.5 | $(build_host_dir)
	ln -f $< $@

$(build_host_dir)/liboffload.so.5: $(obj_liboffload_host) | $(build_host_dir)
	$(compiler_host) $(opts_liboffload_host) $(opts_host) $^ -o $@

$(obj_host_dir)/%.o: $(source_dir)/%.c $(headers) | $(obj_host_dir)
	$(compiler_host) $(opts_common) $(opts_host) $< -o $@

$(obj_host_dir)/%.o: $(source_dir)/%.cpp $(headers) | $(obj_host_dir)
	$(compiler_host) $(opts_common) $(opts_host) $< -o $@

$(obj_host_dir)/%.o: $(source_dir)/coi/%.cpp $(headers) | $(obj_host_dir)
	$(compiler_host) $(opts_common) $(opts_host) $< -o $@

$(obj_host_dir)/%.o: $(source_dir)/orsl-lite/lib/%.c $(headers) | $(obj_host_dir)
	$(compiler_host) $(opts_common) $(opts_host) $< -o $@


$(build_target_dir)/liboffload.so: $(build_target_dir)/liboffload.so.5 | $(build_target_dir)
	ln -f $< $@

$(build_target_dir)/liboffload.so.5: $(obj_liboffload_target) | $(build_target_dir)
	$(compiler_target) $(opts_liboffload_target) $(opts_target) $^ -o $@

$(obj_target_dir)/%.o: $(source_dir)/%.c $(headers) | $(obj_target_dir)
	$(compiler_target) $(opts_common) $(opts_target) $< -o $@

$(obj_target_dir)/%.o: $(source_dir)/%.cpp $(headers) | $(obj_target_dir)
	$(compiler_target) $(opts_common) $(opts_target) $< -o $@

$(obj_target_dir)/%.o: $(source_dir)/coi/%.cpp $(headers) | $(obj_target_dir)
	$(compiler_target) $(opts_common) $(opts_target) $< -o $@

$(obj_target_dir)/%.o: $(source_dir)/orsl-lite/lib/%.c $(headers) | $(obj_target_dir)
	$(compiler_target) $(opts_common) $(opts_target) $< -o $@


$(build_host_dir)/%.o: $(source_dir)/%.cpp $(headers) | $(build_host_dir)
	$(compiler_host) $(opts_common) $(opts_host) $< -o $@

$(build_target_dir)/%.o: $(source_dir)/%.cpp $(headers) | $(build_target_dir)
	$(compiler_target) $(opts_common) $(opts_target) $< -o $@


$(imported_dir)/omp.h: $(omp_header_dir)/omp.h | $(imported_dir)
	cp $< $@


$(build_host_dir) $(build_target_dir) $(obj_host_dir) $(obj_target_dir): | $(build_dir)
	$(shell mkdir -p $@ >/dev/null 2>/dev/null)
	@echo "Created $@ directory"

$(build_dir):
	$(shell mkdir -p $@ >/dev/null 2>/dev/null)
	@echo "Created $@ directory"

$(imported_dir):
	$(shell mkdir -p $@ >/dev/null 2>/dev/null)
	@echo "Created $@ directory"


clean:
	$(shell rm -rf $(build_dir))
	@echo "Remove $(build_dir) directory"


info:
	@echo "root_dir = $(root_dir)"
	@echo "build_dir = $(build_dir)"
	@echo "mpss_dir = $(mpss_dir)"
	@echo "mpss_version = $(mpss_version)"
	@echo "libiomp_host_dir = $(libiomp_host_dir)"
	@echo "libiomp_target_dir = $(libiomp_target_dir)"
	@echo "omp_header_dir = $(omp_header_dir)"
	@echo "os_host = $(os_host)"
	@echo "os_target = $(os_target)"
	@echo "compiler_host = $(compiler_host)"
	@echo "compiler_target = $(compiler_target)"
	@echo "options_host = $(options_host)"
	@echo "options_target = $(options_target)"

