// override-system-header.h to test out 'override' warning.
// rdar://18295240
#define END_COM_MAP virtual unsigned AddRef(void) = 0;

#define STDMETHOD(method)        virtual void method
#define IFACEMETHOD(method)         STDMETHOD(method)
