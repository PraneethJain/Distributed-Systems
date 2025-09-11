// Compile: mpic++ -O3 -march=native -std=c++17 main.cpp

#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;

struct Entry
{
    int col;
    double val;
};
using Row = vector<Entry>;
static constexpr double EPS = 1e-12;

Row read_row(istream &in)
{
    int k;
    if (!(in >> k))
        return {};
    Row r;
    r.reserve(k);
    for (int i = 0; i < k; ++i)
    {
        int c;
        double v;
        in >> c >> v;
        r.push_back({c, v});
    }
    return r;
}

int main(int argc, char **argv)
{
    if (argc > 1)
    {
        if (freopen(argv[1], "r", stdin) == nullptr)
        {
            perror("freopen");
            return 1;
        }
    }

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 0, M = 0, P = 0;
    vector<Row> A_rows;
    vector<int> B_row_ptr;
    vector<int> B_cols;
    vector<double> B_vals;

    // ---------- Rank 0: read & flatten ----------
    if (rank == 0)
    {
        cin >> N >> M >> P;
        A_rows.resize(N);
        for (int i = 0; i < N; i++)
            A_rows[i] = read_row(cin);

        vector<Row> tmpB(M);
        for (int i = 0; i < M; i++)
            tmpB[i] = read_row(cin);

        B_row_ptr.resize(M + 1);
        B_row_ptr[0] = 0;
        for (int i = 0; i < M; i++)
            B_row_ptr[i + 1] = B_row_ptr[i] + (int)tmpB[i].size();

        int tot = B_row_ptr[M];
        B_cols.resize(tot);
        B_vals.resize(tot);
        int idx = 0;
        for (int i = 0; i < M; i++)
            for (auto &e : tmpB[i])
            {
                B_cols[idx] = e.col;
                B_vals[idx] = e.val;
                ++idx;
            }
    }

    // ---------- Broadcast B ----------
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
        B_row_ptr.resize(M + 1);
    MPI_Bcast(B_row_ptr.data(), M + 1, MPI_INT, 0, MPI_COMM_WORLD);

    int totB = B_row_ptr[M];
    if (rank != 0)
    {
        B_cols.resize(totB);
        B_vals.resize(totB);
    }
    MPI_Bcast(B_cols.data(), totB, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B_vals.data(), totB, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ---------- Partition rows of A by nnz ----------
    vector<vector<int>> rowsForRank(size);
    if (rank == 0)
    {
        vector<long long> work(size, 0);
        for (int i = 0; i < N; i++)
        {
            long long nnz = A_rows[i].size();
            int best = (int)(min_element(work.begin(), work.end()) - work.begin());
            rowsForRank[best].push_back(i);
            work[best] += nnz;
        }
    }

    vector<int> sendCounts(size, 0);
    if (rank == 0)
        for (int r = 0; r < size; r++)
            sendCounts[r] = rowsForRank[r].size();

    int myCount;
    MPI_Scatter(sendCounts.data(), 1, MPI_INT, &myCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> myRows(myCount);
    if (rank == 0)
    {
        myRows = rowsForRank[0];
        for (int r = 1; r < size; r++)
            if (!rowsForRank[r].empty())
                MPI_Send(rowsForRank[r].data(), sendCounts[r], MPI_INT, r, 11, MPI_COMM_WORLD);
    }
    else if (myCount > 0)
        MPI_Recv(myRows.data(), myCount, MPI_INT, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // ---------- Ship actual A rows ----------
    vector<Row> myA;
    myA.reserve(myCount);
    if (rank == 0)
    {
        for (int i = 0; i < myCount; i++)
            myA.push_back(move(A_rows[myRows[i]]));
        for (int r = 1; r < size; r++)
        {
            for (int idx = 0; idx < sendCounts[r]; idx++)
            {
                int gi = rowsForRank[r][idx];
                Row &row = A_rows[gi];
                int k = row.size();
                MPI_Send(&k, 1, MPI_INT, r, 20, MPI_COMM_WORLD);
                if (k > 0)
                {
                    vector<int> cc(k);
                    vector<double> vv(k);
                    for (int j = 0; j < k; j++)
                    {
                        cc[j] = row[j].col;
                        vv[j] = row[j].val;
                    }
                    MPI_Send(cc.data(), k, MPI_INT, r, 21, MPI_COMM_WORLD);
                    MPI_Send(vv.data(), k, MPI_DOUBLE, r, 22, MPI_COMM_WORLD);
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < myCount; i++)
        {
            int k;
            MPI_Recv(&k, 1, MPI_INT, 0, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            Row r;
            r.resize(k);
            if (k > 0)
            {
                vector<int> cc(k);
                vector<double> vv(k);
                MPI_Recv(cc.data(), k, MPI_INT, 0, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(vv.data(), k, MPI_DOUBLE, 0, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < k; j++)
                {
                    r[j].col = cc[j];
                    r[j].val = vv[j];
                }
            }
            myA.push_back(move(r));
        }
    }

    // Precompute B row nnz
    vector<int> Bnnz(M);
    for (int i = 0; i < M; i++)
        Bnnz[i] = B_row_ptr[i + 1] - B_row_ptr[i];

    vector<long long> est(myCount, 0);
    for (int i = 0; i < myCount; i++)
        for (auto &a : myA[i])
            if (a.col >= 0 && a.col < M)
                est[i] += Bnnz[a.col];

    // ---------- TIME ONLY PARALLEL MULTIPLY ----------
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    const int DENSE_THR = 512;
    const int P_MAX = 200000;
    vector<double> dense_acc;
    vector<int> dense_touch;
    dense_touch.reserve(1024);
    vector<pair<int, Row>> localC;
    localC.reserve(myCount);

    for (int idx = 0; idx < myCount; idx++)
    {
        int gid = myRows[idx];
        Row &arow = myA[idx];
        if (arow.empty())
        {
            localC.emplace_back(gid, Row{});
            continue;
        }
        bool dense = (P <= P_MAX) && (est[idx] >= DENSE_THR);
        if (dense)
        {
            dense_acc.assign(P, 0.0);
            dense_touch.clear();
            for (auto &a : arow)
            {
                int brow = a.col;
                if (brow < 0 || brow >= M)
                    continue;
                int s = B_row_ptr[brow], e = B_row_ptr[brow + 1];
                for (int j = s; j < e; j++)
                {
                    int col = B_cols[j];
                    double val = a.val * B_vals[j];
                    if (dense_acc[col] == 0)
                        dense_touch.push_back(col);
                    dense_acc[col] += val;
                }
            }
            Row out;
            out.reserve(dense_touch.size());
            for (int c : dense_touch)
                if (fabs(dense_acc[c]) > EPS)
                    out.push_back({c, dense_acc[c]});
            sort(out.begin(), out.end(), [](auto &x, auto &y)
                 { return x.col < y.col; });
            localC.emplace_back(gid, move(out));
        }
        else
        {
            unordered_map<int, double> acc;
            acc.reserve((size_t)min<long long>(est[idx], 1'000'000LL));
            for (auto &a : arow)
            {
                int brow = a.col;
                if (brow < 0 || brow >= M)
                    continue;
                int s = B_row_ptr[brow], e = B_row_ptr[brow + 1];
                for (int j = s; j < e; j++)
                    acc[B_cols[j]] += a.val * B_vals[j];
            }
            Row out;
            out.reserve(acc.size());
            for (auto &kv : acc)
                if (fabs(kv.second) > EPS)
                    out.push_back({kv.first, kv.second});
            sort(out.begin(), out.end(), [](auto &x, auto &y)
                 { return x.col < y.col; });
            localC.emplace_back(gid, move(out));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local = t1 - t0, maxT;
    MPI_Reduce(&local, &maxT, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
        cerr << "ExecutionTime: " << maxT << " sec\n";

    // ---------- Gather C ----------
    if (rank == 0)
    {
        vector<Row> Cres(N);
        for (auto &p : localC)
            Cres[p.first] = move(p.second);
        for (int r = 1; r < size; r++)
        {
            int cnt;
            MPI_Recv(&cnt, 1, MPI_INT, r, 40, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < cnt; i++)
            {
                int rid, k;
                MPI_Recv(&rid, 1, MPI_INT, r, 41, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&k, 1, MPI_INT, r, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                Row rr(k);
                if (k > 0)
                {
                    vector<int> cc(k);
                    vector<double> vv(k);
                    MPI_Recv(cc.data(), k, MPI_INT, r, 43, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(vv.data(), k, MPI_DOUBLE, r, 44, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (int j = 0; j < k; j++)
                    {
                        rr[j].col = cc[j];
                        rr[j].val = vv[j];
                    }
                }
                Cres[rid] = move(rr);
            }
        }
        cout.setf(std::ios::fmtflags(0), std::ios::floatfield);
        cout << setprecision(12);
        for (int i = 0; i < N; i++)
        {
            Row &r = Cres[i];
            cout << r.size();
            for (auto &e : r)
                cout << " " << e.col << " " << e.val;
            cout << "\n";
        }
    }
    else
    {
        int cnt = localC.size();
        MPI_Send(&cnt, 1, MPI_INT, 0, 40, MPI_COMM_WORLD);
        for (auto &p : localC)
        {
            int rid = p.first, k = p.second.size();
            MPI_Send(&rid, 1, MPI_INT, 0, 41, MPI_COMM_WORLD);
            MPI_Send(&k, 1, MPI_INT, 0, 42, MPI_COMM_WORLD);
            if (k > 0)
            {
                vector<int> cc(k);
                vector<double> vv(k);
                for (int i = 0; i < k; i++)
                {
                    cc[i] = p.second[i].col;
                    vv[i] = p.second[i].val;
                }
                MPI_Send(cc.data(), k, MPI_INT, 0, 43, MPI_COMM_WORLD);
                MPI_Send(vv.data(), k, MPI_DOUBLE, 0, 44, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
