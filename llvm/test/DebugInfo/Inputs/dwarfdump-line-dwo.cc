struct foo {
};

foo f;

// Built with GCC
// $ mkdir -p /tmp/dbginfo
// $ cp dwarfdump-line-dwo.cc /tmp/dbginfo
// $ cd /tmp/dbginfo
// $ g++ -c -fdebug-types-section dwarfdump-line-dwo.cc -o <output>
