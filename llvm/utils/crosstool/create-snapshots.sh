#!/bin/bash
#
# Creates LLVM SVN snapshots: llvm-$REV.tar.bz2 and llvm-gcc-4.2-$REV.tar.bz2,
# where $REV is an SVN revision of LLVM.  This is used for creating stable
# tarballs which can be used to build known-to-work crosstools.
#
# Syntax:
#   $0 [REV] -- grabs the revision $REV from SVN; if not specified, grabs the
#   latest SVN revision.

set -o nounset
set -o errexit

readonly REV="${1:-HEAD}"

runOnModule() {
  local module=$1
  local log="${module}.log"
  echo "Running: svn co -r ${REV} ${module}; log in ${log}"
  svn co -r ${REV} http://llvm.org/svn/llvm-project/${module}/trunk ${module} \
      > ${log} 2>&1

  # Delete all the ".svn" dirs; they take quite a lot of space.
  echo "Cleaning up .svn dirs"
  find ${module} -type d -name \.svn -print0 | xargs -0 /bin/rm -rf

  # Create "module-revision.tar.bz2" packages from the SVN checkout dirs.
  local revision=$(grep "Checked out revision" ${log} | \
                   sed 's/[^0-9]\+\([0-9]\+\)[^0-9]\+/\1/')
  local tarball="${module}-${revision}.tar.bz2"
  echo "Creating tarball: ${tarball}"
  tar cjf ${tarball} ${module}

  echo "Cleaning SVN checkout dir ${module}"
  rm -rf ${module} ${log}
}

for module in "llvm" "llvm-gcc-4.2"; do
  runOnModule ${module}
done

