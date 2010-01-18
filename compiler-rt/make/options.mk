# Options which may be overriden for platforms, etc.
#
# This list of such variables should be kept up to date with AvailableOptions in
# 'make/lib_info.mk'.

# The compiler to use.
CC := gcc

# The compiler flags to use.
CFLAGS := -Wall -Werror

# The list of functions to include in the library.
FUNCTIONS :=

# Whether optimized function implementations should be used.
OPTIMIZED := 1

# Miscellaneous tools.

AR := ar
# FIXME: Remove these pipes once ranlib errors are fixed.
ARFLAGS := cru 2> /dev/null
RANLIB := ranlib
# FIXME: Remove these pipes once ranlib errors are fixed.
RANLIBFLAGS := 2> /dev/null
