// override-system-header.h to test out 'override' warning.
// rdar://18295240
#define END_COM_MAP virtual unsigned AddRef(void) = 0;
