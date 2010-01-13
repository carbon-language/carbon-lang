# Makefile utilities

###
# Utility functions

# Function: Set variable value
#
# Set the given make variable to the given value.
Set = $(eval $(1) := $(2))

# Function: Append variable value
#
# Append the given value to the given make variable.
Append = $(eval $(1) += $(2))

###
# Clean up make behavior

# Cancel all suffix rules. We don't want no stinking suffix rules.
.SUFFIXES:

###
# Debugging

# General debugging rule, use 'make print-XXX' to print the definition, value
# and origin of XXX.
make-print-%:
	$(error PRINT: $(value $*) = "$($*)" (from $(origin $*)))

