#ifndef B1_H
#define B1_H
#include "a2.h"
namespace llvm {
class MachineBasicBlock;
template <class NodeT> class DomTreeNodeBase;
typedef DomTreeNodeBase<MachineBasicBlock> MachineDomTreeNode;
}
#endif
