using System;
using System.Diagnostics;
using System.Numerics;
using System.Collections.Generic;
using ILGPU;
using ILGPU.Runtime;

class Program
{
    // ============================
    // Kernel 1: primi distinti <= sqrt(n)
    // ============================
    static void FindPrimeDivisorsKernel(
        Index1D index,
        ArrayView<long> outputPrimes,
        ArrayView<int> outputCount,
        long n)
    {
        long i = index + 2; // candidati 2..sqrt(n)
        if (n % i != 0) return;

        bool isPrime = true;
        for (long j = 2; j * j <= i; j++)
        {
            if (i % j == 0) { isPrime = false; break; }
        }

        if (isPrime)
        {
            // Usa direttamente l'indice restituito dall'add atomico
            int pos = Atomic.Add(ref outputCount[0], 1);
            outputPrimes[pos] = i;
        }
    }


    // ============================
    // Kernel 2: espansione con controllo overflow
    // - expandedFactors: buffer di output
    // - expandedCount: numero scritto
    // - overflowFlag[0] = 1 se overflow
    // ============================
    static void ReduceResidualKernel(
        Index1D _,
        ArrayView<long> primeDivisors,
        ArrayView<int> primeCount,
        ArrayView<long> residual)
    {
        if (_.X != 0) return;

        long r = residual[0];
        int count = primeCount[0];

        for (int i = 0; i < count; i++)
        {
            long p = primeDivisors[i];
            if (p < 2) continue;
            while (r % p == 0) r /= p;
        }

        if (r > 1)
        {
            primeDivisors[count] = r;
            primeCount[0] = count + 1;
        }
        residual[0] = r;
    }

    // ============================
    // Kernel 3: Bitonic sort (ordinamento su GPU)
    // Assume array length = power of two. Per lunghezze arbitrarie, si pad-a con max long.
    // ============================
    static void BitonicCompareAndSwapKernel(
        Index1D idx,
        ArrayView<long> data,
        int j,
        int k)
    {
        int i = idx;
        int ixj = i ^ j;

        if (ixj > i)
        {
            bool ascending = (i & k) == 0;
            long di = data[i];
            long dj = data[ixj];

            if ((ascending && di > dj) || (!ascending && di < dj))
            {
                data[i] = dj;
                data[ixj] = di;
            }
        }
    }

    // ============================
    // Pollard's rho (CPU) per numeri grandi
    // ============================
    // Pollard’s rho (CPU) per numeri grandi
    static long PollardsRho(long n)
    {
        if (n % 2 == 0) return 2;
        var rnd = new Random(42);
        while (true)
        {
            long x = rnd.Next(2, (int)Math.Min(n - 1, int.MaxValue));
            long y = x;
            long c = rnd.Next(1, (int)Math.Min(n - 1, int.MaxValue));
            long d = 1;

            Func<long, long> f = z => (long)((((BigInteger)z * z) + c) % n);

            while (d == 1)
            {
                x = f(x);
                y = f(f(y));
                long diff = Math.Abs(x - y);
                d = (long)BigInteger.GreatestCommonDivisor(diff, n);
                if (d == n) break;
            }

            if (d > 1 && d < n) return d;
        }
    }

    static bool IsPrime(long n)
    {
        if (n < 2) return false;
        if (n % 2 == 0) return n == 2;
        for (long i = 3; i * i <= n; i += 2)
            if (n % i == 0) return false;
        return true;
    }

    static List<long> FactorWithPollards(long n)
    {
        var result = new List<long>();
        var stack = new Stack<long>();
        stack.Push(n);

        while (stack.Count > 0)
        {
            long m = stack.Pop();
            if (m == 1) continue;
            if (IsPrime(m)) { result.Add(m); continue; }
            long d = PollardsRho(m);
            if (d == m)
            {
                for (long i = 2; i * i <= m; i++)
                {
                    if (m % i == 0) { stack.Push(i); stack.Push(m / i); goto next; }
                }
                result.Add(m);
            }
            else
            {
                stack.Push(d);
                stack.Push(m / d);
            }
        next:;
        }

        result.Sort();
        return result;
    }

    // ============================
    // Utility: round-up to power of two
    // ============================
    static int NextPowerOfTwo(int x)
    {
        if (x <= 1) return 1;
        x--;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return x + 1;
    }

    static void Main()
    {
        // Creazione del contesto ILGPU
        using var context = Context.CreateDefault();
        // Selezione del device preferito (GPU se disponibile, altrimenti CPU)
        var device = context.GetPreferredDevice(preferCPU: false);
        // Creazione dell'acceleratore associato al device scelto
        using var accelerator = device.CreateAccelerator(context);

        // Caricamento dei kernel GPU
        var findKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<long>, ArrayView<int>, long>(FindPrimeDivisorsKernel);

        var reduceKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<long>, ArrayView<int>, ArrayView<long>>(ReduceResidualKernel);

        var bitonicKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<long>, int, int>(BitonicCompareAndSwapKernel);

        Console.WriteLine($"Uso acceleratore: {accelerator}");
        Console.WriteLine("Premi ESC per uscire.");

        // Loop infinito per gestire input multipli
        while (true)
        {
            Console.Write("\nInserisci un numero >= 2: ");
            string input = "";
            // Lettura carattere per carattere per intercettare ESC immediatamente
            while (true)
            {
                var key = Console.ReadKey(true);
                if (key.Key == ConsoleKey.Escape) Environment.Exit(0); // chiusura immediata
                else if (key.Key == ConsoleKey.Enter) { Console.WriteLine(); break; }
                else { input += key.KeyChar; Console.Write(key.KeyChar); }
            }

            // Validazione input
            if (!long.TryParse(input, out long n) || n < 2)
            {
                Console.WriteLine("Numero non valido.");
                continue;
            }

            var sw = Stopwatch.StartNew(); // cronometro per misurare tempo

            // Soglia: se il numero è molto grande, usa Pollard’s rho su CPU
            const long PollardThreshold = 1_000_000_000;
            if (n > PollardThreshold)
            {
                var factors = FactorWithPollards(n);
                var distinct = new SortedSet<long>(factors);
                Console.WriteLine($"Fattori primi di {n}: {string.Join(", ", distinct)}");
                Console.WriteLine($"Tempo impiegato: {sw.ElapsedMilliseconds} ms");
                continue;
            }

            // Limite di ricerca: sqrt(n)
            int limit = (int)Math.Sqrt(n);
            using var primeDivisorsBuf = accelerator.Allocate1D<long>(limit + 1);
            using var primeCountBuf = accelerator.Allocate1D<int>(1);

            // Azzeramento buffer per evitare valori spuri
            primeDivisorsBuf.MemSetToZero();
            primeCountBuf.CopyFromCPU(new int[] { 0 });

            // Kernel 1: trova i primi distinti <= sqrt(n)
            findKernel(limit - 1, primeDivisorsBuf.View, primeCountBuf.View, n);
            accelerator.Synchronize();

            // Kernel 2: riduzione residuo e aggiunta eventuale primo > sqrt(n)
            using var residualBuf = accelerator.Allocate1D<long>(1);
            residualBuf.CopyFromCPU(new long[] { n });
            reduceKernel(1, primeDivisorsBuf.View, primeCountBuf.View, residualBuf.View);
            accelerator.Synchronize();

            // Recupero numero di primi trovati e array host
            int primeDistinctCount = primeCountBuf.GetAsArray1D()[0];
            long[] primesHost = primeDivisorsBuf.GetAsArray1D();

            // Ordinamento su GPU con Bitonic sort
            int pow2 = NextPowerOfTwo(Math.Max(1, primeDistinctCount)); // padding a potenza di 2
            using var sortBuf = accelerator.Allocate1D<long>(pow2);
            long[] toSort = new long[pow2];
            for (int i = 0; i < pow2; i++)
                toSort[i] = i < primeDistinctCount ? primesHost[i] : long.MaxValue;
            sortBuf.CopyFromCPU(toSort);

            // Esecuzione Bitonic sort
            for (int k = 2; k <= pow2; k <<= 1)
            {
                for (int j = k >> 1; j > 0; j >>= 1)
                {
                    bitonicKernel(pow2, sortBuf.View, j, k);
                    accelerator.Synchronize();
                }
            }

            // Recupero risultati ordinati
            long[] sortedAll = sortBuf.GetAsArray1D();
            var sortedDistinct = new List<long>();
            for (int i = 0; i < pow2; i++)
            {
                long v = sortedAll[i];
                if (v == long.MaxValue) break; // ignora padding
                if (v > 1) sortedDistinct.Add(v); // filtro: niente 0 o valori spuri
            }

            sw.Stop(); // stop cronometro

            // Stampa risultati
            if (sortedDistinct.Count == 0)
                Console.WriteLine($"Fattori primi di {n}: {n}");
            else
                Console.WriteLine($"Fattori primi di {n}: {string.Join(", ", sortedDistinct)}");

            Console.WriteLine($"Tempo impiegato: {sw.ElapsedMilliseconds} ms");
        }
    }
}
