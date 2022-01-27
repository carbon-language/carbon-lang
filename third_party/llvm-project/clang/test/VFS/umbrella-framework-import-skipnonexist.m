// REQUIRES: crash-recovery

// RUN: rm -rf %t
// RUN: mkdir -p %t/vdir %t/outdir %t/cache
// RUN: cp -R %S/Inputs/Bar.framework %t/outdir/
//
// RUN: sed -e "s@VDIR@%{/t:regex_replacement}/vdir@g" -e "s@OUT_DIR@%{/t:regex_replacement}/outdir@g" %S/Inputs/bar-headers.yaml > %t/vdir/bar-headers.yaml
// RUN: rm -f %t/outdir/Bar.framework/Headers/B.h
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -ivfsoverlay %t/vdir/bar-headers.yaml -F %t/vdir -fsyntax-only %s

@import Bar;
