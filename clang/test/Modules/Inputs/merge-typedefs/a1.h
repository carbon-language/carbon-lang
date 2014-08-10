#ifndef A1_H
#define A1_H
namespace llvm {
class MachineBasicBlock;
template <class NodeT> class DomTreeNodeBase;
typedef DomTreeNodeBase<MachineBasicBlock> MachineDomTreeNode;
}
#endif
