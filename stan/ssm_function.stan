functions {
  matrix kronecker_prod(matrix A, matrix B) {
    int m;
    int n;
    int p;
    int q;
    m = rows(A);
    n = cols(A);
    p = rows(B);
    q = cols(B);
    matrix[m * p, n * q] C;
    
    for (i in 1:m) {
      for (j in 1:n) {
        int row_start;
        int row_end;
        int col_start;
        int col_end;
        row_start = (i - 1) * p + 1;
        row_end = (i - 1) * p + p;
        col_start = (j - 1) * q + 1;
        col_end = (j - 1) * q + q;
        C[row_start:row_end, col_start:col_end] = A[i, j] * B;
      }
    }
    return C;
  }
  
  vector to_vector_lower_tri(matrix A) {
    int n;
    int k;
    n = rows(A);
    k = 1;
    vector[n * (n + 1) %/% 2] v;
    
    for (c in 1:n) {
      for (r in c:n) {
        v[k] = A[r, c];
        k = k + 1;
      }
    }
    return v;
  }

  int index_of_diag_lower_tri(int i, int n){
    array[n, n] int index_mat = rep_array(0, n, n);
    int m = n * (n + 1) %/% 2;
    array[m] int index_vec = linspaced_int_array(m, 1, m);
    int k = 1;
    
    for (c in 1:n) {
      for (r in c:n) {
        index_mat[r, c] = index_vec[k];
        k = k + 1;
      }
    }
    
    return index_mat[i, i];
  }
}
