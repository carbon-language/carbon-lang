#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/DataTypes.h"
#include "gtest/gtest.h"
#include <climits>

using namespace llvm;

namespace {

TEST(BlockFrequencyTest, OneToZero) {
  BlockFrequency Freq(1);
  BranchProbability Prob(UINT32_MAX - 1, UINT32_MAX);
  Freq *= Prob;
  EXPECT_EQ(Freq.getFrequency(), 0u);

  Freq = BlockFrequency(1);
  uint32_t Remainder = Freq.scale(Prob);
  EXPECT_EQ(Freq.getFrequency(), 0u);
  EXPECT_EQ(Remainder, UINT32_MAX - 1);
}

TEST(BlockFrequencyTest, OneToOne) {
  BlockFrequency Freq(1);
  BranchProbability Prob(UINT32_MAX, UINT32_MAX);
  Freq *= Prob;
  EXPECT_EQ(Freq.getFrequency(), 1u);

  Freq = BlockFrequency(1);
  uint32_t Remainder = Freq.scale(Prob);
  EXPECT_EQ(Freq.getFrequency(), 1u);
  EXPECT_EQ(Remainder, 0u);
}

TEST(BlockFrequencyTest, ThreeToOne) {
  BlockFrequency Freq(3);
  BranchProbability Prob(3000000, 9000000);
  Freq *= Prob;
  EXPECT_EQ(Freq.getFrequency(), 1u);

  Freq = BlockFrequency(3);
  uint32_t Remainder = Freq.scale(Prob);
  EXPECT_EQ(Freq.getFrequency(), 1u);
  EXPECT_EQ(Remainder, 0u);
}

TEST(BlockFrequencyTest, MaxToHalfMax) {
  BlockFrequency Freq(UINT64_MAX);
  BranchProbability Prob(UINT32_MAX / 2, UINT32_MAX);
  Freq *= Prob;
  EXPECT_EQ(Freq.getFrequency(), 9223372034707292159ULL);

  Freq = BlockFrequency(UINT64_MAX);
  uint32_t Remainder = Freq.scale(Prob);
  EXPECT_EQ(Freq.getFrequency(), 9223372034707292159ULL);
  EXPECT_EQ(Remainder, 0u);
}

TEST(BlockFrequencyTest, BigToBig) {
  const uint64_t Big = 387246523487234346LL;
  const uint32_t P = 123456789;
  BlockFrequency Freq(Big);
  BranchProbability Prob(P, P);
  Freq *= Prob;
  EXPECT_EQ(Freq.getFrequency(), Big);

  Freq = BlockFrequency(Big);
  uint32_t Remainder = Freq.scale(Prob);
  EXPECT_EQ(Freq.getFrequency(), Big);
  EXPECT_EQ(Remainder, 0u);
}

TEST(BlockFrequencyTest, MaxToMax) {
  BlockFrequency Freq(UINT64_MAX);
  BranchProbability Prob(UINT32_MAX, UINT32_MAX);
  Freq *= Prob;
  EXPECT_EQ(Freq.getFrequency(), UINT64_MAX);

  // This additionally makes sure if we have a value equal to our saturating
  // value, we do not signal saturation if the result equals said value, but
  // saturating does not occur.
  Freq = BlockFrequency(UINT64_MAX);
  uint32_t Remainder = Freq.scale(Prob);
  EXPECT_EQ(Freq.getFrequency(), UINT64_MAX);
  EXPECT_EQ(Remainder, 0u);
}

TEST(BlockFrequencyTest, ScaleResultRemainderTest) {
  struct {
    uint64_t Freq;
    uint32_t Prob[2];
    uint64_t ExpectedFreq;
    uint32_t ExpectedRemainder;
  } Tests[80] = {
    // Data for scaling that results in <= 64 bit division.
    { 0x1423e2a50, { 0x64819521, 0x7765dd13 }, 0x10f418889, 0x92b9d25 },
    { 0x35ef14ce, { 0x28ade3c7, 0x304532ae }, 0x2d73c33a, 0x2c0fd0b6 },
    { 0xd03dbfbe24, { 0x790079, 0xe419f3 }, 0x6e776fc1fd, 0x4a06dd },
    { 0x21d67410b, { 0x302a9dc2, 0x3ddb4442 }, 0x1a5948fd6, 0x265d1c2a },
    { 0x8664aead, { 0x3d523513, 0x403523b1 }, 0x805a04cf, 0x324c27b8 },
    { 0x201db0cf4, { 0x35112a7b, 0x79fc0c74 }, 0xdf8b07f6, 0x490c1dc4 },
    { 0x13f1e4430a, { 0x21c92bf, 0x21e63aae }, 0x13e0cba15, 0x1df47c30 },
    { 0x16c83229, { 0x3793f66f, 0x53180dea }, 0xf3ce7b6, 0x1d0c1b6b },
    { 0xc62415be8, { 0x9cc4a63, 0x4327ae9b }, 0x1ce8b71ca, 0x3f2c696a },
    { 0x6fac5e434, { 0xe5f9170, 0x1115e10b }, 0x5df23dd4c, 0x4dafc7c },
    { 0x1929375f2, { 0x3a851375, 0x76c08456 }, 0xc662b082, 0x343589ee },
    { 0x243c89db6, { 0x354ebfc0, 0x450ef197 }, 0x1bf8c1661, 0x4948e49 },
    { 0x310e9b31a, { 0x1b1b8acf, 0x2d3629f0 }, 0x1d69c93f9, 0x73e3b96 },
    { 0xa1fae921d, { 0xa7a098c, 0x10469f44 }, 0x684413d6c, 0x86a882c },
    { 0xc1582d957, { 0x498e061, 0x59856bc }, 0x9edc5f4e7, 0x29b0653 },
    { 0x57cfee75, { 0x1d061dc3, 0x7c8bfc17 }, 0x1476a220, 0x2383d33f },
    { 0x139220080, { 0x294a6c71, 0x2a2b07c9 }, 0x1329e1c76, 0x7aa5da },
    { 0x1665d353c, { 0x7080db5, 0xde0d75c }, 0xb590d9fb, 0x7ba8c38 },
    { 0xe8f14541, { 0x5188e8b2, 0x736527ef }, 0xa4971be5, 0x6b612167 },
    { 0x2f4775f29, { 0x254ef0fe, 0x435fcf50 }, 0x1a2e449c1, 0x28bbf5e },
    { 0x27b85d8d7, { 0x304c8220, 0x5de678f2 }, 0x146e3bef9, 0x4b27097e },
    { 0x1d362e36b, { 0x36c85b12, 0x37a66f55 }, 0x1cc19b8e6, 0x688e828 },
    { 0x155fd48c7, { 0xf5894d, 0x1256108 }, 0x11e383602, 0x111f0cb },
    { 0xb5db2d15, { 0x39bb26c5, 0x5bdcda3e }, 0x72499259, 0x59c4939b },
    { 0x153990298, { 0x48921c09, 0x706eb817 }, 0xdb3268e8, 0x66bb8a80 },
    { 0x28a7c3ed7, { 0x1f776fd7, 0x349f7a70 }, 0x184f73ae1, 0x28910321 },
    { 0x724dbeab, { 0x1bd149f5, 0x253a085e }, 0x5569c0b3, 0xff8e2ed },
    { 0xd8f0c513, { 0x18c8cc4c, 0x1b72bad0 }, 0xc3e30643, 0xd85e134 },
    { 0x17ce3dcb, { 0x1e4c6260, 0x233b359e }, 0x1478f4af, 0x49ea31e },
    { 0x1ce036ce0, { 0x29e3c8af, 0x5318dd4a }, 0xe8e76196, 0x11d5b9c4 },
    { 0x1473ae2a, { 0x29b897ba, 0x2be29378 }, 0x13718185, 0x6f93b2c },
    { 0x1dd41aa68, { 0x3d0a4441, 0x5a0e8f12 }, 0x1437b6bbf, 0x54b09ffa },
    { 0x1b49e4a53, { 0x3430c1fe, 0x5a204aed }, 0xfcd6852f, 0x15ad6ed7 },
    { 0x217941b19, { 0x12ced2bd, 0x21b68310 }, 0x12aca65b1, 0x1b2a9565 },
    { 0xac6a4dc8, { 0x3ed68da8, 0x6fdca34c }, 0x60da926d, 0x22ff53e4 },
    { 0x1c503a4e7, { 0xfcbbd32, 0x11e48d17 }, 0x18fec7d38, 0xa8aa816 },
    { 0x1c885855, { 0x213e919d, 0x25941897 }, 0x193de743, 0x4ea09c },
    { 0x29b9c168e, { 0x2b644aea, 0x45725ee7 }, 0x1a122e5d5, 0xbee1099 },
    { 0x806a33f2, { 0x30a80a23, 0x5063733a }, 0x4db9a264, 0x1eaed76e },
    { 0x282afc96b, { 0x143ae554, 0x1a9863ff }, 0x1e8de5204, 0x158d9020 },
    // Data for scaling that results in > 64 bit division.
    { 0x23ca5f2f672ca41c, { 0xecbc641, 0x111373f7 }, 0x1f0301e5e8295ab5, 0xf627f79 },
    { 0x5e4f2468142265e3, { 0x1ddf5837, 0x32189233 }, 0x383ca7ba9fdd2c8c, 0x1c8f33e1 },
    { 0x277a1a6f6b266bf6, { 0x415d81a8, 0x61eb5e1e }, 0x1a5a3e1d41b30c0f, 0x29cde3ae },
    { 0x1bdbb49a237035cb, { 0xea5bf17, 0x1d25ffb3 }, 0xdffc51c53d44b93, 0x5170574 },
    { 0x2bce6d29b64fb8, { 0x3bfd5631, 0x7525c9bb }, 0x166ebedda7ac57, 0x3026dfab },
    { 0x3a02116103df5013, { 0x2ee18a83, 0x3299aea8 }, 0x35be8922ab1e2a84, 0x298d9919 },
    { 0x7b5762390799b18c, { 0x12f8e5b9, 0x2563bcd4 }, 0x3e960077aca01209, 0x93afeb8 },
    { 0x69cfd72537021579, { 0x4c35f468, 0x6a40feee }, 0x4be4cb3848be98a3, 0x4ff96b9e },
    { 0x49dfdf835120f1c1, { 0x8cb3759, 0x559eb891 }, 0x79663f7120edade, 0x51b1fb5b },
    { 0x74b5be5c27676381, { 0x47e4c5e0, 0x7c7b19ff }, 0x4367d2dff36a1028, 0x7a7b5608 },
    { 0x4f50f97075e7f431, { 0x9a50a17, 0x11cd1185 }, 0x2af952b34c032df4, 0xfddc6a3 },
    { 0x2f8b0d712e393be4, { 0x1487e386, 0x15aa356e }, 0x2d0df36478a776aa, 0x14e2564c },
    { 0x224c1c75999d3de, { 0x3b2df0ea, 0x4523b100 }, 0x1d5b481d145f08a, 0x15145eec },
    { 0x2bcbcea22a399a76, { 0x28b58212, 0x48dd013e }, 0x187814d084c47cab, 0x3a38ebe2 },
    { 0x1dbfca91257cb2d1, { 0x1a8c04d9, 0x5e92502c }, 0x859cf7d00f77545, 0x7431f4d },
    { 0x7f20039b57cda935, { 0xeccf651, 0x323f476e }, 0x25720cd976461a77, 0x202817a3 },
    { 0x40512c6a586aa087, { 0x113b0423, 0x398c9eab }, 0x1341c03de8696a7e, 0x1e27284b },
    { 0x63d802693f050a11, { 0xf50cdd6, 0xfce2a44 }, 0x60c0177bb5e46846, 0xf7ad89e },
    { 0x2d956b422838de77, { 0xb2d345b, 0x1321e557 }, 0x1aa0ed16b6aa5319, 0xfe1a5ce },
    { 0x5a1cdf0c1657bc91, { 0x1d77bb0c, 0x1f991ff1 }, 0x54097ee94ff87560, 0x11c4a26c },
    { 0x3801b26d7e00176b, { 0xeed25da, 0x1a819d8b }, 0x1f89e96a3a639526, 0xcd51e7c },
    { 0x37655e74338e1e45, { 0x300e170a, 0x5a1595fe }, 0x1d8cfb55fddc0441, 0x3df05434 },
    { 0x7b38703f2a84e6, { 0x66d9053, 0xc79b6b9 }, 0x3f7d4c91774094, 0x26d939e },
    { 0x2245063c0acb3215, { 0x30ce2f5b, 0x610e7271 }, 0x113b916468389235, 0x1b588512 },
    { 0x6bc195877b7b8a7e, { 0x392004aa, 0x4a24e60c }, 0x530594fb17db6ba5, 0x35c0a5f0 },
    { 0x40a3fde23c7b43db, { 0x4e712195, 0x6553e56e }, 0x320a799bc76a466a, 0x5e23a5eb },
    { 0x1d3dfc2866fbccba, { 0x5075b517, 0x5fc42245 }, 0x18917f0061595bc3, 0x3fcf4527 },
    { 0x19aeb14045a61121, { 0x1bf6edec, 0x707e2f4b }, 0x6626672a070bcc7, 0x3607801f },
    { 0x44ff90486c531e9f, { 0x66598a, 0x8a90dc }, 0x32f6f2b0525199b0, 0x5ab576 },
    { 0x3f3e7121092c5bcb, { 0x1c754df7, 0x5951a1b9 }, 0x14267f50b7ef375d, 0x221220a8 },
    { 0x60e2dafb7e50a67e, { 0x4d96c66e, 0x65bd878d }, 0x49e31715ac393f8b, 0x4e97b195 },
    { 0x656286667e0e6e29, { 0x9d971a2, 0xacda23b }, 0x5c6ee315ead6cb4f, 0x516f5bd },
    { 0x1114e0974255d507, { 0x1c693, 0x2d6ff }, 0xaae42e4b35f6e60, 0x8b65 },
    { 0x508c8baf3a70ff5a, { 0x3b26b779, 0x6ad78745 }, 0x2c98387636c4b365, 0x11dc6a51 },
    { 0x5b47bc666bf1f9cf, { 0x10a87ed6, 0x187d358a }, 0x3e1767155848368b, 0xfb871c },
    { 0x50954e3744460395, { 0x7a42263, 0xcdaa048 }, 0x2fe739f0aee1fee1, 0xb8add57 },
    { 0x20020b406550dd8f, { 0x3318539, 0x42eead0 }, 0x186f326325fa346b, 0x10d3ae7 },
    { 0x5bcb0b872439ffd5, { 0x6f61fb2, 0x9af7344 }, 0x41fa1e3bec3c1b30, 0x4fee45a },
    { 0x7a670f365db87a53, { 0x417e102, 0x3bb54c67 }, 0x8642a558304fd9e, 0x3b65f514 },
    { 0x1ef0db1e7bab1cd0, { 0x2b60cf38, 0x4188f78f }, 0x147ae0d6226b2ee6, 0x336b6106 }
  };

  for (unsigned i = 0; i < 80; i++) {
    BlockFrequency Freq(Tests[i].Freq);
    uint32_t Remainder = Freq.scale(BranchProbability(Tests[i].Prob[0],
                                                      Tests[i].Prob[1]));
    EXPECT_EQ(Tests[i].ExpectedFreq, Freq.getFrequency());
    EXPECT_EQ(Tests[i].ExpectedRemainder, Remainder);
  }
}

TEST(BlockFrequency, Divide) {
  BlockFrequency Freq(0x3333333333333333ULL);
  Freq /= BranchProbability(1, 2);
  EXPECT_EQ(Freq.getFrequency(), 0x6666666666666666ULL);
}

TEST(BlockFrequencyTest, Saturate) {
  BlockFrequency Freq(0x3333333333333333ULL);
  Freq /= BranchProbability(100, 300);
  EXPECT_EQ(Freq.getFrequency(), 0x9999999999999999ULL);
  Freq /= BranchProbability(1, 2);
  EXPECT_EQ(Freq.getFrequency(), UINT64_MAX);

  Freq = 0x1000000000000000ULL;
  Freq /= BranchProbability(10000, 160000);
  EXPECT_EQ(Freq.getFrequency(), UINT64_MAX);

  // Try to cheat the multiplication overflow check.
  Freq = 0x00000001f0000001ull;
  Freq /= BranchProbability(1000, 0xf000000f);
  EXPECT_EQ(33506781356485509ULL, Freq.getFrequency());
}

TEST(BlockFrequencyTest, ProbabilityCompare) {
  BranchProbability A(4, 5);
  BranchProbability B(4U << 29, 5U << 29);
  BranchProbability C(3, 4);

  EXPECT_TRUE(A == B);
  EXPECT_FALSE(A != B);
  EXPECT_FALSE(A < B);
  EXPECT_FALSE(A > B);
  EXPECT_TRUE(A <= B);
  EXPECT_TRUE(A >= B);

  EXPECT_FALSE(B == C);
  EXPECT_TRUE(B != C);
  EXPECT_FALSE(B < C);
  EXPECT_TRUE(B > C);
  EXPECT_FALSE(B <= C);
  EXPECT_TRUE(B >= C);

  BranchProbability BigZero(0, UINT32_MAX);
  BranchProbability BigOne(UINT32_MAX, UINT32_MAX);
  EXPECT_FALSE(BigZero == BigOne);
  EXPECT_TRUE(BigZero != BigOne);
  EXPECT_TRUE(BigZero < BigOne);
  EXPECT_FALSE(BigZero > BigOne);
  EXPECT_TRUE(BigZero <= BigOne);
  EXPECT_FALSE(BigZero >= BigOne);
}

}
