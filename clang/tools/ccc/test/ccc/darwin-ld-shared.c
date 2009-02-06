// -shared translates to -dynamiclib on darwin.
// RUN: xcc -ccc-host-system darwin -### -filelist a &> %t.1 &&
// RUN: xcc -ccc-host-system darwin -### -filelist a -shared &> %t.2 &&

// -dynamiclib turns on -dylib
// RUN: not grep -- '-dylib' %t.1 &&
// RUN: grep -- '-dylib' %t.2
