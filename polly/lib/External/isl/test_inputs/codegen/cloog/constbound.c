for (int c0 = 0; c0 <= 199; c0 += 1) {
  for (int c2 = 50 * c0; c2 <= 50 * c0 + 24; c2 += 1)
    for (int c3 = 0; c3 <= c2; c3 += 1)
      S1(c0, c2, c3);
  for (int c2 = 50 * c0 + 25; c2 <= 50 * c0 + 49; c2 += 1)
    for (int c3 = 0; c3 <= c2; c3 += 1)
      S2(c0, c2, c3);
}
