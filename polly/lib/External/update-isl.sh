#! /bin/sh
set -e

# Replace the content of the isl directory with a fresh clone from
# http://repo.or.cz/isl.git

SCRIPTPATH=`realpath --no-symlinks $(dirname $0)`
ISL_SOURCE_DIR="${SCRIPTPATH}/isl"

TMPDIR=`mktemp -d --tmpdir isl-XXX`
GITDIR=$TMPDIR/src
BUILDDIR=$TMPDIR/build

git clone --recursive http://repo.or.cz/isl.git $GITDIR
if [ -n "$1" ]; then
  (cd $GITDIR && git checkout $1)
  (cd $GITDIR && git submodule update --recursive)
fi
(cd $GITDIR && ./autogen.sh)
mkdir -p $BUILDDIR
(cd $BUILDDIR && $GITDIR/configure --with-int=imath-32 --with-clang=system)
echo "#define GIT_HEAD_ID \"\"" > $GITDIR/gitversion.h
(cd $BUILDDIR && make -j dist)

for DISTFILE in "$BUILDDIR/isl*.tar.gz"; do break; done

cp $ISL_SOURCE_DIR/include/isl/isl-noexceptions.h $TMPDIR/isl-noexceptions.h

rm -rf $ISL_SOURCE_DIR
mkdir -p $ISL_SOURCE_DIR
tar -xf $DISTFILE --strip-components=1 --directory $ISL_SOURCE_DIR
cp $TMPDIR/isl-noexceptions.h $ISL_SOURCE_DIR/include/isl

rm -rf $TMPDIR
