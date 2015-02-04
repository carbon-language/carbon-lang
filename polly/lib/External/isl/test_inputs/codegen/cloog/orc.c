{
  for (int c1 = 0; c1 <= 2; c1 += 1) {
    S1(c1);
    for (int c2 = 0; c2 <= -c1 + 11; c2 += 1) {
      S2(c1, c2);
      S3(c1, c2);
    }
    S4(c1);
  }
  for (int c1 = 0; c1 <= 14; c1 += 1) {
    S5(c1);
    for (int c2 = 0; c2 <= 9; c2 += 1)
      S6(c1, c2);
    S7(c1);
  }
}
