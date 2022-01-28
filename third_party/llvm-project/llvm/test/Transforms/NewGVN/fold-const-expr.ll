; GVN failed to do constant expression folding and expanded
; them unfolded in many places, producing exponentially large const
; expressions. As a result, the compilation never fisished.
; This test checks that we are folding constant expression
; PR 28418
; RUN: opt -newgvn -S < %s | FileCheck %s
%2 = type { i32, i32, i32, i32, i32 }
define i32 @_Z16vector3util_mainv(i32 %x, i32 %y)  {
  %tmp1 = alloca %2, align 4
  %tmp114 = getelementptr inbounds %2, %2* %tmp1, i64 0, i32 1
  %tmp115 = bitcast i32* %tmp114 to <4 x i32>*
  store <4 x i32> <i32 234567891, i32 345678912, i32 456789123, i32 0>, <4 x i32>* %tmp115, align 4
  %tmp1683 = getelementptr inbounds %2, %2* %tmp1, i64 0, i32 1
  %tmp1688 = load i32, i32* %tmp1683, align 4
  %tmp1693 = shl i32 %tmp1688, 5
  %tmp1694 = xor i32 %tmp1693, %tmp1688
  %tmp1695 = lshr i32 %tmp1694, 7
  %tmp1696 = xor i32 %tmp1695, %tmp1694
  %tmp1697 = shl i32 %tmp1696, 22
  %tmp1698 = xor i32 %tmp1697, %tmp1696
  %tmp1707 = shl i32 %tmp1698, 5
  %tmp1708 = xor i32 %tmp1707, %tmp1698
  %tmp1709 = lshr i32 %tmp1708, 7
  %tmp1710 = xor i32 %tmp1709, %tmp1708
  %tmp1711 = shl i32 %tmp1710, 22
  %tmp1712 = xor i32 %tmp1711, %tmp1710
  %tmp1721 = shl i32 %tmp1712, 5
  %tmp1722 = xor i32 %tmp1721, %tmp1712
  %tmp1723 = lshr i32 %tmp1722, 7
  %tmp1724 = xor i32 %tmp1723, %tmp1722
  %tmp1725 = shl i32 %tmp1724, 22
  %tmp1726 = xor i32 %tmp1725, %tmp1724
  %tmp1735 = shl i32 %tmp1726, 5
  %tmp1736 = xor i32 %tmp1735, %tmp1726
  %tmp1737 = lshr i32 %tmp1736, 7
  %tmp1738 = xor i32 %tmp1737, %tmp1736
  %tmp1739 = shl i32 %tmp1738, 22
  %tmp1740 = xor i32 %tmp1739, %tmp1738
  store i32 %tmp1740, i32* %tmp1683, align 4
; CHECK: store i32 310393545, i32* %tmp114, align 4
  %tmp1756 = getelementptr inbounds %2, %2* %tmp1, i64 0, i32 1
  %tmp1761 = load i32, i32* %tmp1756, align 4
  %tmp1766 = shl i32 %tmp1761, 5
  %tmp1767 = xor i32 %tmp1766, %tmp1761
  %tmp1768 = lshr i32 %tmp1767, 7
  %tmp1769 = xor i32 %tmp1768, %tmp1767
  %tmp1770 = shl i32 %tmp1769, 22
  %tmp1771 = xor i32 %tmp1770, %tmp1769
  %tmp1780 = shl i32 %tmp1771, 5
  %tmp1781 = xor i32 %tmp1780, %tmp1771
  %tmp1782 = lshr i32 %tmp1781, 7
  %tmp1783 = xor i32 %tmp1782, %tmp1781
  %tmp1784 = shl i32 %tmp1783, 22
  %tmp1785 = xor i32 %tmp1784, %tmp1783
  %tmp1794 = shl i32 %tmp1785, 5
  %tmp1795 = xor i32 %tmp1794, %tmp1785
  %tmp1796 = lshr i32 %tmp1795, 7
  %tmp1797 = xor i32 %tmp1796, %tmp1795
  %tmp1798 = shl i32 %tmp1797, 22
  %tmp1799 = xor i32 %tmp1798, %tmp1797
  %tmp1808 = shl i32 %tmp1799, 5
  %tmp1809 = xor i32 %tmp1808, %tmp1799
  %tmp1810 = lshr i32 %tmp1809, 7
  %tmp1811 = xor i32 %tmp1810, %tmp1809
  %tmp1812 = shl i32 %tmp1811, 22
  %tmp1813 = xor i32 %tmp1812, %tmp1811
  store i32 %tmp1813, i32* %tmp1756, align 4
; CHECK: store i32 -383584258, i32* %tmp114, align 4
  %tmp2645 = getelementptr inbounds %2, %2* %tmp1, i64 0, i32 1
  %tmp2650 = load i32, i32* %tmp2645, align 4
  %tmp2655 = shl i32 %tmp2650, 5
  %tmp2656 = xor i32 %tmp2655, %tmp2650
  %tmp2657 = lshr i32 %tmp2656, 7
  %tmp2658 = xor i32 %tmp2657, %tmp2656
  %tmp2659 = shl i32 %tmp2658, 22
  %tmp2660 = xor i32 %tmp2659, %tmp2658
  %tmp2669 = shl i32 %tmp2660, 5
  %tmp2670 = xor i32 %tmp2669, %tmp2660
  %tmp2671 = lshr i32 %tmp2670, 7
  %tmp2672 = xor i32 %tmp2671, %tmp2670
  %tmp2673 = shl i32 %tmp2672, 22
  %tmp2674 = xor i32 %tmp2673, %tmp2672
  %tmp2683 = shl i32 %tmp2674, 5
  %tmp2684 = xor i32 %tmp2683, %tmp2674
  %tmp2685 = lshr i32 %tmp2684, 7
  %tmp2686 = xor i32 %tmp2685, %tmp2684
  %tmp2687 = shl i32 %tmp2686, 22
  %tmp2688 = xor i32 %tmp2687, %tmp2686
  %tmp2697 = shl i32 %tmp2688, 5
  %tmp2698 = xor i32 %tmp2697, %tmp2688
  %tmp2699 = lshr i32 %tmp2698, 7
  %tmp2700 = xor i32 %tmp2699, %tmp2698
  %tmp2701 = shl i32 %tmp2700, 22
  %tmp2702 = xor i32 %tmp2701, %tmp2700
  store i32 %tmp2702, i32* %tmp2645, align 4
; CHECK: store i32 -57163022, i32* %tmp114, align 4
  ret i32 0
}
