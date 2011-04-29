#!/bin/bash
#
# get_cloog.sh - retrieve cloog from git repo to current directory, configure, build
# and install into <destdir>, which defaults to cwd.
#
# Basic process is:
# 1. git clone git://repo.or.cz/cloog.git
# 2. cd cloog
# 3. ./get_submodules.sh
# 4. ./autogen.sh
# 5. ./configure --prefix=<destdir>
#

this=$0
verb=false
destDir=none
libGMPDir=/usr
forceClone=false
cmdPre=""

usage() {
  echo usage: $this '[--verbose --dry-run --force --gmp=<gmpdir> <install-directory>]'
  echo 'where: <install-directory> must exist, defaults to cwd'
  echo '       <gmpdir>: path to GMP library, defaults to /usr'
  exit $1
}

vsay() {
  if [[ $verb = true ]] ; then
	echo "$@"
  fi
}

complain() {
  echo "$@"
  exit 1
}

# Function to make a directory including its ancestors.
makedir() {
  if [[ $1 == "" ]] ; then
	:
  elif [[ -d $1 ]] ; then
	:
  else
    makedirtmp=`dirname "$1"`
    makedir "$makedirtmp"
    vsay "$this: Making directory $1"
    if mkdir "$1" ; then
      :
    else
      complain "$this: mkdir $1 failed"
    fi
  fi
}

# echo/run command line, complain if it fails.
runCmd() {
  $cmdPre $*
  if [[ $? != 0 ]] ; then
    complain $* failed
  fi
}

ACTUAL_PWD=`pwd`
ROOT=${PWD:-$ACTUAL_PWD}
vsay $this: Polylibs root is $ROOT.

while [[ $# != 0 ]] ; do
  tmp=`echo $1 | sed -e 's;^[-]*;;g'`
  if [[ $tmp == $1 ]] ; then
    if [[ $destDir == none ]] ; then
      destDir=$tmp
    fi
  else
    switchArg=`echo $tmp | cut -s -d= -f2`
    if [[ $switchArg != "" ]] ; then
      tmp=`echo $tmp | cut -d= -f1`
    fi
    case $tmp in
    dry-run)
      cmdPre=echo
      ;;
    force)
      forceClone=true
      ;;
    gmp)
      if [[ $switchArg == "" ]] ; then
        echo "$this: --gmp requires a pathname argument"
        usage 1
      fi
      libGMPDir=$switchArg
      ;;
    help)
      usage 0
      ;;
    verbose)
      verb=true
      ;;
    *)
      usage 1
      ;;
    esac
  fi
  shift
done

if [[ $destDir == none ]] ; then
  destDir=$ACTUAL_PWD
else
  # Create specified install directory if it doesn't exist.
  makedir $destDir
fi

destDir=`(cd $destDir ; echo $PWD)`

if [[ ! -d $libGMPDir ]] ; then
  complain GMP dir $libGMPDIR not found.
fi
if [[ ! -f $libGMPDir/include/gmp.h ]] ; then
  complain $libGMPDir does not appear to contain a GMP library.
fi

# Check for libtoolize, required by cloog/autogen.sh
tmp=`which libtoolize`
if [[ $tmp == "" ]] ; then
  complain libtoolize '(needed by cloog/autogen.sh)' not found
fi

# 1. Get cloog, remove existing cloog tree if forcing.
if [[ $forceClone == true ]] ; then
  vsay removing existing cloog tree
  rm -rf ./cloog
fi

# Allow reuse of existing cloog tree.
if [[ ! -d cloog ]] ; then
  vsay cloning git repo into $ROOT/cloog
  runCmd git clone git://repo.or.cz/cloog.git
else
  vsay using existing cloog tree: $ROOT/cloog
fi

runCmd cd cloog

# 2. Get bundled isl library
runCmd ./get_submodules.sh

# 3. Generate configure scripts for cloog and isl
runCmd ./autogen.sh

configArgs="--without-polylib --with-isl=bundled"
configArgs="$configArgs --with-gmp-prefix=$libGMPDir --prefix=$destDir"

# 4. Configure cloog and isl
runCmd ./configure $configArgs

exit 0


