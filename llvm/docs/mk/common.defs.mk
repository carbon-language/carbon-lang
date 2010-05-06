##===- docs/mk/common.defs.mk ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

# support for printing errors, warnings and info.

# FUNCTION: print error message once and dump makefile list.
# $1: error message. do NOT end in punctuation.
# $2: optional. one or more words to list. if words > 1 then 1 per line.
define fn.DUMP.error
$(if $(filter 0,$(words $2)) \
     ,$(error $(call _DUMP.error,$1.)) \
     ,$(if $(filter 1,$(words $2)) \
           ,$(error $(call _DUMP.error,$1: $2)) \
           ,$(error $(call _DUMP.errorm,$1:,$2))))
endef

# FUNCTION: print one error message per word and dump makefile list.
# $1: error message. do NOT end in punctuation.
# $2: one or more words to list. 1 word per line.
define fn.DUMP.errorn
$(error $(call _DUMP.errorn,$1,$2))
endef

##===----------------------------------------------------------------------===##

# MAKEFILE PRIVATE.
# support for special makefile printing. Whitespace is very important for all
# definitions; modify with caution.

define _DUMP.error
ERROR.
***
*** $1
***
$(call _DUMP.makefile_list,***,$1) ERROR
endef

define _DUMP.errorm
ERROR.
***
*** $1
***$(subst *** ,***,$(foreach n,$2,    $n$(_TEXT.blank)***))
$(call _DUMP.makefile_list,***,$1) ERROR
endef

define _DUMP.errorn
ERROR.
***
***$(subst *** ,***,$(foreach n,$2, ERROR: $1: $n$(_TEXT.blank)***))
$(call _DUMP.makefile_list,***,$1) ERROR
endef

define _DUMP.makefile_list
$1 MAKEFILE_LIST:
$1     $(abspath $(word 1,$(MAKEFILE_LIST)))
$1$(subst $()$1 ,$1,$(foreach n,$(wordlist 2,999,$(MAKEFILE_LIST)),    $n$(_TEXT.blank)$1))
$1
endef

# force linefeed.
define _TEXT.blank


endef

##===----------------------------------------------------------------------===##

# FUNCTION: select
# $1: variable basename (before .__.)
# $2: selection key
# RESULT: variable selection by key
#
SELECT = $(if $($1.__.$2),$($1.__.$2),$($1.__.default))

# FUNCTION: safe(r) rm -rf
# $1: list of dirs to remove, recursively
#
# Extra safety is put in place by wrapping dirs in $(realpath) which both cleans
# up the path and returns blank if the path does not exist.
#
# An error is produced if removal of any generally unsafe dirs is attempted.
# 
fn.RMRF = $(if $(call _RMRF.eval,$1) \
  ,$(call fn.DUMP.errorn,unsafe recursive dir removal in call function fn.RMRF,$(call _RMRF.eval,$1)) \
  ,$(RM) -rf $(realpath $1))

_RMRF.eval = $(sort $(realpath $(filter $(realpath $(_RMRF.never)),$(realpath $1))))

_RMRF.never = / /bin /dev /etc /home /net /opt /sbin /tmp /usr /var \
              /usr/bin /usr/sbin /usr/share \
              /usr/local/bin /usr/local/sbin /usr/local/share

# FUNCTION: pipe and compress.
# $1: target filename with filename extension of compression, if any.
# RESULT: a suitable pipe for compression or empty.
#
fn.PIPE.COMP = $(fn.PIPE.COMP.$(suffix $1))
fn.PIPE.COMP..gz  = | $(GZIP)
fn.PIPE.COMP..bz2 = | $(BZIP2)

# FUNCTION: pipe and decompress.
# $1: target filename with filename extension of compression, if any.
# RESULT: a suitable pipe for decompression or empty.
#
fn.PIPE.DECOMP = $(fn.PIPE.DECOMP.$(suffix $1))
fn.PIPE.DECOMP..gz  = | $(GZIP) -d
fn.PIPE.DECOMP..bz2 = | $(BZIP2) -d

# FUNCTION: fetch working copy 'Last Changed Rev'.
# $1: working copy directory.
# RESULT: 'Last Changed Rev' value from svn info.
#
fn.SVN.LCREV = $(shell (svn info $1 2>/dev/null || echo "Last Changed Rev: 0") | $(SED) -n 's/^Last Changed Rev: *\([0-9][0-9]*\)/\1/p')
