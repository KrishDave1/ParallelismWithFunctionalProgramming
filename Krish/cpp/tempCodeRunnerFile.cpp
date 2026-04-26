double sequentialMonteCarloPi(int n, unsigned int seed) {
    mt19937 gen(seed);
    uniform_real_distribution<double> dist(0.0, 1.0);
    int hits = 0;
    for (int i = 0; i < n; i++) {
        double x = dist(gen);
        double y = dist(gen);
        if (x * x + y * y <= 1.0) hits++;
    }
    return 4.0 * hits / n;
}