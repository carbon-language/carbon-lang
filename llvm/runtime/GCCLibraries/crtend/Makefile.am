#===-- runtime/GCCLibraries/crtend/Makefile.am -------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

##===- runtime/GCCLibraries/crtend/Makefile ----------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# This directory contains the C and C++ runtime libraries for the LLVM GCC
# front-ends.  See the README.txt file for more details.
#
# Since this archive has strange requirements, we use almost all custom rules
# for building it.
#
##===----------------------------------------------------------------------===##

include $(top_srcdir)/Makefile_config

AggregateLib  := $(BCDIR)/libcrtend.a
ComponentLibs := comp_main.bc comp_genericeh.bc comp_sjljeh.bc

## We build libcrtend.a from the four components described in the README.
$(AggregateLib) : $(BCDIR) $(ComponentLibs)
	@echo Building final libcrtend.a file from bytecode components
	$(AR) cr $(BCDIR)/libcrtend.a $(ComponentLibs)

all-am: $(AggregateLib)

clean-am: clean-crtend

clean-crtend:
	$(RM) -f $(AggregateLib) *.bc

install-am: install-crtend

install-crtend:
	$(INSTALL_DATA) '$(AggregateLib)' $(DESTDIR)$(libdir)

MainObj      := crtend.bc listend.bc
GenericEHObj := Exception.bc
SJLJEHObj    := SJLJ-Exception.bc

# __main and ctor/dtor support component
comp_main.bc: $(MainObj)
	@echo Linking $(notdir $@) component...
	$(LLVMGCCLD) -link-as-library -internalize-public-api-file=$(srcdir)/comp_main.lst $(MainObj) -o $@

# Generic exception handling support runtime.
comp_genericeh.bc: $(GenericEHObj)
	@echo Linking $(notdir $@) component...
	$(LLVMGCCLD) -link-as-library -internalize-public-api-file=$(srcdir)/comp_genericeh.lst $(GenericEHObj) -o $@

# setjmp/longjmp exception handling support runtime.
comp_sjljeh.bc: $(SJLJEHObj)
	@echo Linking $(notdir $@) component...
	$(LLVMGCCLD) -link-as-library -internalize-public-api-file=$(srcdir)/comp_sjljeh.lst $(SJLJEHObj) -o $@
