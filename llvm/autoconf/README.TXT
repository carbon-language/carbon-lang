Upgrading Libtool
-----------------

If you are in the mood to upgrade libtool, you must do the following:

1. configure/build/install libtool with --prefix=<PFX>
2. Copy <PFX>/share/aclocal/libtool.m4 to llvm/autoconf/m4/libtool.m4
3. Copy <PFX>/share/libtool/ltmain.sh to llvm/autoconf/ltmain.sh
4. Adjust the llvm/autoconf/m4/libtool.m4 file to ensure that the variable
   default_ofile is set to "mklib" instead of "libtool"
5. Possibly adjust llvm/autoconf/m4/libtool.m4 to get rid of autoconf 
   warnings related to obsolete autconf features (e.g. AC_TRY_LINK)
