//===- unittest/Support/BranchProbabilityTest.cpp - BranchProbability tests -=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace llvm {
void PrintTo(const BranchProbability &P, ::std::ostream *os) {
  *os << P.getNumerator() << "/" << P.getDenominator();
}
}
namespace {

typedef BranchProbability BP;
TEST(BranchProbabilityTest, Accessors) {
  EXPECT_EQ(1u, BP(1, 7).getNumerator());
  EXPECT_EQ(7u, BP(1, 7).getDenominator());
  EXPECT_EQ(0u, BP::getZero().getNumerator());
  EXPECT_EQ(1u, BP::getZero().getDenominator());
  EXPECT_EQ(1u, BP::getOne().getNumerator());
  EXPECT_EQ(1u, BP::getOne().getDenominator());
}

TEST(BranchProbabilityTest, Operators) {
  EXPECT_TRUE(BP(1, 7) < BP(2, 7));
  EXPECT_TRUE(BP(1, 7) < BP(1, 4));
  EXPECT_TRUE(BP(5, 7) < BP(3, 4));
  EXPECT_FALSE(BP(1, 7) < BP(1, 7));
  EXPECT_FALSE(BP(1, 7) < BP(2, 14));
  EXPECT_FALSE(BP(4, 7) < BP(1, 2));
  EXPECT_FALSE(BP(4, 7) < BP(3, 7));

  EXPECT_FALSE(BP(1, 7) > BP(2, 7));
  EXPECT_FALSE(BP(1, 7) > BP(1, 4));
  EXPECT_FALSE(BP(5, 7) > BP(3, 4));
  EXPECT_FALSE(BP(1, 7) > BP(1, 7));
  EXPECT_FALSE(BP(1, 7) > BP(2, 14));
  EXPECT_TRUE(BP(4, 7) > BP(1, 2));
  EXPECT_TRUE(BP(4, 7) > BP(3, 7));

  EXPECT_TRUE(BP(1, 7) <= BP(2, 7));
  EXPECT_TRUE(BP(1, 7) <= BP(1, 4));
  EXPECT_TRUE(BP(5, 7) <= BP(3, 4));
  EXPECT_TRUE(BP(1, 7) <= BP(1, 7));
  EXPECT_TRUE(BP(1, 7) <= BP(2, 14));
  EXPECT_FALSE(BP(4, 7) <= BP(1, 2));
  EXPECT_FALSE(BP(4, 7) <= BP(3, 7));

  EXPECT_FALSE(BP(1, 7) >= BP(2, 7));
  EXPECT_FALSE(BP(1, 7) >= BP(1, 4));
  EXPECT_FALSE(BP(5, 7) >= BP(3, 4));
  EXPECT_TRUE(BP(1, 7) >= BP(1, 7));
  EXPECT_TRUE(BP(1, 7) >= BP(2, 14));
  EXPECT_TRUE(BP(4, 7) >= BP(1, 2));
  EXPECT_TRUE(BP(4, 7) >= BP(3, 7));

  EXPECT_FALSE(BP(1, 7) == BP(2, 7));
  EXPECT_FALSE(BP(1, 7) == BP(1, 4));
  EXPECT_FALSE(BP(5, 7) == BP(3, 4));
  EXPECT_TRUE(BP(1, 7) == BP(1, 7));
  EXPECT_TRUE(BP(1, 7) == BP(2, 14));
  EXPECT_FALSE(BP(4, 7) == BP(1, 2));
  EXPECT_FALSE(BP(4, 7) == BP(3, 7));

  EXPECT_TRUE(BP(1, 7) != BP(2, 7));
  EXPECT_TRUE(BP(1, 7) != BP(1, 4));
  EXPECT_TRUE(BP(5, 7) != BP(3, 4));
  EXPECT_FALSE(BP(1, 7) != BP(1, 7));
  EXPECT_FALSE(BP(1, 7) != BP(2, 14));
  EXPECT_TRUE(BP(4, 7) != BP(1, 2));
  EXPECT_TRUE(BP(4, 7) != BP(3, 7));
}

TEST(BranchProbabilityTest, getCompl) {
  EXPECT_EQ(BP(5, 7), BP(2, 7).getCompl());
  EXPECT_EQ(BP(2, 7), BP(5, 7).getCompl());
  EXPECT_EQ(BP::getZero(), BP(7, 7).getCompl());
  EXPECT_EQ(BP::getOne(), BP(0, 7).getCompl());
}

TEST(BranchProbabilityTest, scale) {
  // Multiply by 1.0.
  EXPECT_EQ(UINT64_MAX, BP(1, 1).scale(UINT64_MAX));
  EXPECT_EQ(UINT64_MAX, BP(7, 7).scale(UINT64_MAX));
  EXPECT_EQ(UINT32_MAX, BP(1, 1).scale(UINT32_MAX));
  EXPECT_EQ(UINT32_MAX, BP(7, 7).scale(UINT32_MAX));
  EXPECT_EQ(0u, BP(1, 1).scale(0));
  EXPECT_EQ(0u, BP(7, 7).scale(0));

  // Multiply by 0.0.
  EXPECT_EQ(0u, BP(0, 1).scale(UINT64_MAX));
  EXPECT_EQ(0u, BP(0, 1).scale(UINT64_MAX));
  EXPECT_EQ(0u, BP(0, 1).scale(0));

  auto Two63 = UINT64_C(1) << 63;
  auto Two31 = UINT64_C(1) << 31;

  // Multiply by 0.5.
  EXPECT_EQ(Two63 - 1, BP(1, 2).scale(UINT64_MAX));

  // Big fractions.
  EXPECT_EQ(1u, BP(Two31, UINT32_MAX).scale(2));
  EXPECT_EQ(Two31, BP(Two31, UINT32_MAX).scale(Two31 * 2));
  EXPECT_EQ(Two63 + Two31, BP(Two31, UINT32_MAX).scale(UINT64_MAX));

  // High precision.
  EXPECT_EQ(UINT64_C(9223372047592194055),
            BP(Two31 + 1, UINT32_MAX - 2).scale(UINT64_MAX));
}

TEST(BranchProbabilityTest, scaleByInverse) {
  // Divide by 1.0.
  EXPECT_EQ(UINT64_MAX, BP(1, 1).scaleByInverse(UINT64_MAX));
  EXPECT_EQ(UINT64_MAX, BP(7, 7).scaleByInverse(UINT64_MAX));
  EXPECT_EQ(UINT32_MAX, BP(1, 1).scaleByInverse(UINT32_MAX));
  EXPECT_EQ(UINT32_MAX, BP(7, 7).scaleByInverse(UINT32_MAX));
  EXPECT_EQ(0u, BP(1, 1).scaleByInverse(0));
  EXPECT_EQ(0u, BP(7, 7).scaleByInverse(0));

  // Divide by something very small.
  EXPECT_EQ(UINT64_MAX, BP(1, UINT32_MAX).scaleByInverse(UINT64_MAX));
  EXPECT_EQ(uint64_t(UINT32_MAX) * UINT32_MAX,
            BP(1, UINT32_MAX).scaleByInverse(UINT32_MAX));
  EXPECT_EQ(UINT32_MAX, BP(1, UINT32_MAX).scaleByInverse(1));

  auto Two63 = UINT64_C(1) << 63;
  auto Two31 = UINT64_C(1) << 31;

  // Divide by 0.5.
  EXPECT_EQ(UINT64_MAX - 1, BP(1, 2).scaleByInverse(Two63 - 1));
  EXPECT_EQ(UINT64_MAX, BP(1, 2).scaleByInverse(Two63));

  // Big fractions.
  EXPECT_EQ(1u, BP(Two31, UINT32_MAX).scaleByInverse(1));
  EXPECT_EQ(2u, BP(Two31 - 1, UINT32_MAX).scaleByInverse(1));
  EXPECT_EQ(Two31 * 2 - 1, BP(Two31, UINT32_MAX).scaleByInverse(Two31));
  EXPECT_EQ(Two31 * 2 + 1, BP(Two31 - 1, UINT32_MAX).scaleByInverse(Two31));
  EXPECT_EQ(UINT64_MAX, BP(Two31, UINT32_MAX).scaleByInverse(Two63 + Two31));

  // High precision.  The exact answers to these are close to the successors of
  // the floor.  If we were rounding, these would round up.
  EXPECT_EQ(UINT64_C(18446744065119617030),
            BP(Two31 + 2, UINT32_MAX - 2)
                .scaleByInverse(UINT64_C(9223372047592194055)));
  EXPECT_EQ(UINT64_C(18446744065119617026),
            BP(Two31 + 1, UINT32_MAX).scaleByInverse(Two63 + Two31));
}

TEST(BlockProbabilityTest, scaleBruteForce) {
  struct {
    uint64_t Num;
    uint32_t Prob[2];
    uint64_t Result;
  } Tests[] = {
    // Data for scaling that results in <= 64 bit division.
    { 0x1423e2a50ULL, { 0x64819521, 0x7765dd13 }, 0x10f418889ULL },
    { 0x35ef14ceULL, { 0x28ade3c7, 0x304532ae }, 0x2d73c33aULL },
    { 0xd03dbfbe24ULL, { 0x790079, 0xe419f3 }, 0x6e776fc1fdULL },
    { 0x21d67410bULL, { 0x302a9dc2, 0x3ddb4442 }, 0x1a5948fd6ULL },
    { 0x8664aeadULL, { 0x3d523513, 0x403523b1 }, 0x805a04cfULL },
    { 0x201db0cf4ULL, { 0x35112a7b, 0x79fc0c74 }, 0xdf8b07f6ULL },
    { 0x13f1e4430aULL, { 0x21c92bf, 0x21e63aae }, 0x13e0cba15ULL },
    { 0x16c83229ULL, { 0x3793f66f, 0x53180dea }, 0xf3ce7b6ULL },
    { 0xc62415be8ULL, { 0x9cc4a63, 0x4327ae9b }, 0x1ce8b71caULL },
    { 0x6fac5e434ULL, { 0xe5f9170, 0x1115e10b }, 0x5df23dd4cULL },
    { 0x1929375f2ULL, { 0x3a851375, 0x76c08456 }, 0xc662b082ULL },
    { 0x243c89db6ULL, { 0x354ebfc0, 0x450ef197 }, 0x1bf8c1661ULL },
    { 0x310e9b31aULL, { 0x1b1b8acf, 0x2d3629f0 }, 0x1d69c93f9ULL },
    { 0xa1fae921dULL, { 0xa7a098c, 0x10469f44 }, 0x684413d6cULL },
    { 0xc1582d957ULL, { 0x498e061, 0x59856bc }, 0x9edc5f4e7ULL },
    { 0x57cfee75ULL, { 0x1d061dc3, 0x7c8bfc17 }, 0x1476a220ULL },
    { 0x139220080ULL, { 0x294a6c71, 0x2a2b07c9 }, 0x1329e1c76ULL },
    { 0x1665d353cULL, { 0x7080db5, 0xde0d75c }, 0xb590d9fbULL },
    { 0xe8f14541ULL, { 0x5188e8b2, 0x736527ef }, 0xa4971be5ULL },
    { 0x2f4775f29ULL, { 0x254ef0fe, 0x435fcf50 }, 0x1a2e449c1ULL },
    { 0x27b85d8d7ULL, { 0x304c8220, 0x5de678f2 }, 0x146e3bef9ULL },
    { 0x1d362e36bULL, { 0x36c85b12, 0x37a66f55 }, 0x1cc19b8e6ULL },
    { 0x155fd48c7ULL, { 0xf5894d, 0x1256108 }, 0x11e383602ULL },
    { 0xb5db2d15ULL, { 0x39bb26c5, 0x5bdcda3e }, 0x72499259ULL },
    { 0x153990298ULL, { 0x48921c09, 0x706eb817 }, 0xdb3268e8ULL },
    { 0x28a7c3ed7ULL, { 0x1f776fd7, 0x349f7a70 }, 0x184f73ae1ULL },
    { 0x724dbeabULL, { 0x1bd149f5, 0x253a085e }, 0x5569c0b3ULL },
    { 0xd8f0c513ULL, { 0x18c8cc4c, 0x1b72bad0 }, 0xc3e30643ULL },
    { 0x17ce3dcbULL, { 0x1e4c6260, 0x233b359e }, 0x1478f4afULL },
    { 0x1ce036ce0ULL, { 0x29e3c8af, 0x5318dd4a }, 0xe8e76196ULL },
    { 0x1473ae2aULL, { 0x29b897ba, 0x2be29378 }, 0x13718185ULL },
    { 0x1dd41aa68ULL, { 0x3d0a4441, 0x5a0e8f12 }, 0x1437b6bbfULL },
    { 0x1b49e4a53ULL, { 0x3430c1fe, 0x5a204aed }, 0xfcd6852fULL },
    { 0x217941b19ULL, { 0x12ced2bd, 0x21b68310 }, 0x12aca65b1ULL },
    { 0xac6a4dc8ULL, { 0x3ed68da8, 0x6fdca34c }, 0x60da926dULL },
    { 0x1c503a4e7ULL, { 0xfcbbd32, 0x11e48d17 }, 0x18fec7d38ULL },
    { 0x1c885855ULL, { 0x213e919d, 0x25941897 }, 0x193de743ULL },
    { 0x29b9c168eULL, { 0x2b644aea, 0x45725ee7 }, 0x1a122e5d5ULL },
    { 0x806a33f2ULL, { 0x30a80a23, 0x5063733a }, 0x4db9a264ULL },
    { 0x282afc96bULL, { 0x143ae554, 0x1a9863ff }, 0x1e8de5204ULL },
    // Data for scaling that results in > 64 bit division.
    { 0x23ca5f2f672ca41cULL, { 0xecbc641, 0x111373f7 }, 0x1f0301e5e8295ab5ULL },
    { 0x5e4f2468142265e3ULL, { 0x1ddf5837, 0x32189233 }, 0x383ca7ba9fdd2c8cULL },
    { 0x277a1a6f6b266bf6ULL, { 0x415d81a8, 0x61eb5e1e }, 0x1a5a3e1d41b30c0fULL },
    { 0x1bdbb49a237035cbULL, { 0xea5bf17, 0x1d25ffb3 }, 0xdffc51c53d44b93ULL },
    { 0x2bce6d29b64fb8ULL, { 0x3bfd5631, 0x7525c9bb }, 0x166ebedda7ac57ULL },
    { 0x3a02116103df5013ULL, { 0x2ee18a83, 0x3299aea8 }, 0x35be8922ab1e2a84ULL },
    { 0x7b5762390799b18cULL, { 0x12f8e5b9, 0x2563bcd4 }, 0x3e960077aca01209ULL },
    { 0x69cfd72537021579ULL, { 0x4c35f468, 0x6a40feee }, 0x4be4cb3848be98a3ULL },
    { 0x49dfdf835120f1c1ULL, { 0x8cb3759, 0x559eb891 }, 0x79663f7120edadeULL },
    { 0x74b5be5c27676381ULL, { 0x47e4c5e0, 0x7c7b19ff }, 0x4367d2dff36a1028ULL },
    { 0x4f50f97075e7f431ULL, { 0x9a50a17, 0x11cd1185 }, 0x2af952b34c032df4ULL },
    { 0x2f8b0d712e393be4ULL, { 0x1487e386, 0x15aa356e }, 0x2d0df36478a776aaULL },
    { 0x224c1c75999d3deULL, { 0x3b2df0ea, 0x4523b100 }, 0x1d5b481d145f08aULL },
    { 0x2bcbcea22a399a76ULL, { 0x28b58212, 0x48dd013e }, 0x187814d084c47cabULL },
    { 0x1dbfca91257cb2d1ULL, { 0x1a8c04d9, 0x5e92502c }, 0x859cf7d00f77545ULL },
    { 0x7f20039b57cda935ULL, { 0xeccf651, 0x323f476e }, 0x25720cd976461a77ULL },
    { 0x40512c6a586aa087ULL, { 0x113b0423, 0x398c9eab }, 0x1341c03de8696a7eULL },
    { 0x63d802693f050a11ULL, { 0xf50cdd6, 0xfce2a44 }, 0x60c0177bb5e46846ULL },
    { 0x2d956b422838de77ULL, { 0xb2d345b, 0x1321e557 }, 0x1aa0ed16b6aa5319ULL },
    { 0x5a1cdf0c1657bc91ULL, { 0x1d77bb0c, 0x1f991ff1 }, 0x54097ee94ff87560ULL },
    { 0x3801b26d7e00176bULL, { 0xeed25da, 0x1a819d8b }, 0x1f89e96a3a639526ULL },
    { 0x37655e74338e1e45ULL, { 0x300e170a, 0x5a1595fe }, 0x1d8cfb55fddc0441ULL },
    { 0x7b38703f2a84e6ULL, { 0x66d9053, 0xc79b6b9 }, 0x3f7d4c91774094ULL },
    { 0x2245063c0acb3215ULL, { 0x30ce2f5b, 0x610e7271 }, 0x113b916468389235ULL },
    { 0x6bc195877b7b8a7eULL, { 0x392004aa, 0x4a24e60c }, 0x530594fb17db6ba5ULL },
    { 0x40a3fde23c7b43dbULL, { 0x4e712195, 0x6553e56e }, 0x320a799bc76a466aULL },
    { 0x1d3dfc2866fbccbaULL, { 0x5075b517, 0x5fc42245 }, 0x18917f0061595bc3ULL },
    { 0x19aeb14045a61121ULL, { 0x1bf6edec, 0x707e2f4b }, 0x6626672a070bcc7ULL },
    { 0x44ff90486c531e9fULL, { 0x66598a, 0x8a90dc }, 0x32f6f2b0525199b0ULL },
    { 0x3f3e7121092c5bcbULL, { 0x1c754df7, 0x5951a1b9 }, 0x14267f50b7ef375dULL },
    { 0x60e2dafb7e50a67eULL, { 0x4d96c66e, 0x65bd878d }, 0x49e31715ac393f8bULL },
    { 0x656286667e0e6e29ULL, { 0x9d971a2, 0xacda23b }, 0x5c6ee315ead6cb4fULL },
    { 0x1114e0974255d507ULL, { 0x1c693, 0x2d6ff }, 0xaae42e4b35f6e60ULL },
    { 0x508c8baf3a70ff5aULL, { 0x3b26b779, 0x6ad78745 }, 0x2c98387636c4b365ULL },
    { 0x5b47bc666bf1f9cfULL, { 0x10a87ed6, 0x187d358a }, 0x3e1767155848368bULL },
    { 0x50954e3744460395ULL, { 0x7a42263, 0xcdaa048 }, 0x2fe739f0aee1fee1ULL },
    { 0x20020b406550dd8fULL, { 0x3318539, 0x42eead0 }, 0x186f326325fa346bULL },
    { 0x5bcb0b872439ffd5ULL, { 0x6f61fb2, 0x9af7344 }, 0x41fa1e3bec3c1b30ULL },
    { 0x7a670f365db87a53ULL, { 0x417e102, 0x3bb54c67 }, 0x8642a558304fd9eULL },
    { 0x1ef0db1e7bab1cd0ULL, { 0x2b60cf38, 0x4188f78f }, 0x147ae0d6226b2ee6ULL }
  };

  for (const auto &T : Tests) {
    EXPECT_EQ(T.Result, BP(T.Prob[0], T.Prob[1]).scale(T.Num));
  }
}

}
