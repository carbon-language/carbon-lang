LLD_LEVEL := ../../..

BUILT_SOURCES = Version.inc

TABLEGEN_INC_FILES_COMMON = 1

include $(LLD_LEVEL)/Makefile

# Compute the lld version from the LLVM version, unless specified explicitly.
ifndef LLD_VERSION
LLD_VERSION := $(subst svn,,$(LLVMVersion))
LLD_VERSION := $(subst rc,,$(LLD_VERSION))
endif

LLD_VERSION_COMPONENTS := $(subst ., ,$(LLD_VERSION))
LLD_VERSION_MAJOR := $(word 1,$(LLD_VERSION_COMPONENTS))
LLD_VERSION_MINOR := $(word 2,$(LLD_VERSION_COMPONENTS))

LLD_REVISION := $(strip \
        $(shell $(LLVM_SRC_ROOT)/utils/GetSourceVersion $(LLVM_SRC_ROOT)/tools/lld))

LLD_REPOSITORY := $(strip \
        $(shell $(LLVM_SRC_ROOT)/utils/GetRepositoryPath $(LLVM_SRC_ROOT)/tools/lld))

$(ObjDir)/Version.inc.tmp : Version.inc.in Makefile $(LLVM_OBJ_ROOT)/Makefile.config $(ObjDir)/.dir
	$(Echo) "Updating LLD version info."
	$(Verb)sed -e "s#@LLD_VERSION@#$(LLD_VERSION)#g" \
	           -e "s#@LLD_VERSION_MAJOR@#$(LLD_VERSION_MAJOR)#g" \
	           -e "s#@LLD_VERSION_MINOR@#$(LLD_VERSION_MINOR)#g" \
	           -e "s#@LLD_REVISION@#$(LLD_REVISION)#g" \
	           -e "s#@LLD_REPOSITORY@#$(LLD_REPOSITORY)#g" \
	           $< > $@
