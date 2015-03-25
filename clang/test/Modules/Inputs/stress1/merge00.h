#ifndef STRESS1_MERGE00_H
#define STRESS1_MERGE00_H

#include "m00.h"
#include "m01.h"
#include "m02.h"
#include "m03.h"

inline int g() { return N00::S00('a').method00('b') + (int)N00::S00(42) + function00(42); }

#endif
