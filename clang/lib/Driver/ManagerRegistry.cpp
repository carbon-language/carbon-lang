#include "clang/Driver/ManagerRegistry.h"

using namespace clang;

StoreManagerCreator ManagerRegistry::StoreMgrCreator = 0;

ConstraintManagerCreator ManagerRegistry::ConstraintMgrCreator = 0;
