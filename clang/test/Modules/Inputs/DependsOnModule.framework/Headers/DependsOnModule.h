#include <Module/Module.h> //expected-warning{{treating #include as an import of module 'Module'}}

#define DEPENDS_ON_MODULE 1
#private DEPENDS_ON_MODULE

