Upgrading Libtool
===============================================================================

If you are in the mood to upgrade libtool, you must do the following:

 1. Get the new version of libtool and put it in <SRC>
 2. configure/build/install libtool with --prefix=<PFX>
 3. Copy <SRC>/ltdl.m4 to llvm/autoconf/m4
 4. Copy <PFX>/share/aclocal/libtool.m4 to llvm/autoconf/m4/libtool.m4
 5. Copy <PFX>/share/libtool/ltmain.sh to llvm/autoconf/ltmain.sh
 6. Copy <PFX>/share/libtool/libltdl/ltdl.c to llvm/lib/System
 7. Copy <PFX>/share/libtool/libltdl/ltdl.h to llvm/lib/System
 8. Adjust the llvm/autoconf/m4/libtool.m4 file to ensure that the variable
    default_ofile is set to "mklib" instead of "libtool"
 9. Run "autoupdate libtool.m4 ltdl.m4" in the llvm/autoconf/m4 directory.
    This should correctly update the macro definitions in the libtool m4 
    files to match the version of autoconf that LLVM uses. This converts
    AC_HELP_STRING to AS_HELP_STRING and AC_TRY_LINK to AC_LINK_IFELSE, amongst
    other things. You may need to manually adjust the files.
10. Run AutoRegen.sh to get the new macros into configure script
11. If there are any warnings from configure, go to step 9.
12. Rebuild LLVM, making sure it reconfigured
13. Test the JIT which uses libltdl
14. If it all works, only THEN commit the changes.

Upgrading autoconf
===============================================================================

If you are in the mood to upgrade autoconf, you should:

 1. Consider not upgrading.
 2. No really, this is a hassle, you don't want to do it.
 3. Get the new version of autoconf and put it in <SRC>
 4. configure/build/install autoconf with --prefix=<PFX>
 5. Run autoupdate on all the m4 macros in llvm/autoconf/m4
 6. Run autoupdate on llvm/autoconf/configure.ac
 7. Regenerate configure script with AutoRegen.sh
 8. If there are any warnings from autoconf, fix them and go to step 7.
 9. Test, test, test.
