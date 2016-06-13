# from scipy.misc import comb
import codecs
import sys
from os.path import isfile, join
import ast



def calMatrices(mydict):
    mat = []
    b = []
    a = []
    n = 0
    conv_mat = []
    for key in mydict:
        l = []
        for val in mydict[key]:
            l.append(mydict[key][val])

        a.append(sum(l))
        mat.append(l)
        conv_mat.append([key] + l + [sum(l)])
        n += 1

    
    for i in range(0, n):
        suma = 0
        for j in range(0, n):
            suma += mat[j][i]
        b.append(suma)
    conv_mat.append(['Total'] + b + [sum(b)])
    header =  ['class'] + ['T'+str(i) for i in range(0, n)] + ['Total']
    # print '\t'.join(str(elem) for elem in header)
    # for line in conv_mat:
    #     print '\t'.join(str(elem) for elem in line)
    return mat, a, b

def printMatrix(mat):
    format_mat = "{"
    i = 0
    n = len(mat)
    m = len(mat[0])
    for line in mat:
        format_mat += "{"
        j = 0
        for cell in line:
            if j != m-1:
                format_mat += str(cell)+".0,"
            else:
                format_mat += str(cell)+".0"
            j += 1
        if i != n-1:
            format_mat += "},"
        else:
            format_mat += "}"
        i += 1
    format_mat += "}"
    return format_mat

alg = {
    0: 'NMF COUNT time',
    1: 'NMF TFIDF time',
    2: 'KMEANS COUNT time',
    3: 'KMEANS TFIDF time',
    4: 'LDA GENSIM COUNT time',
    5: 'LDA GENSIM TFIDF time',
    6: 'LDA SKLEARN COUNT time',
    7: 'LDA SKLEARN TFIDF time',
    8: 'LSI GENSIM COUNT time',
    9: 'LSI GENSIM TFIDF time'
}

d = {
    0: 'nmf_count',
    1: 'nmf_tfidf',
    2: 'kmean_count',
    3: 'kmean_tfidf',
    4: 'lda_count_gensim',
    5: 'lda_tfidf_gensim',
    6: 'lda_count_sklearn',
    7: 'lda_tfidf_sklearn',
    8: 'lsi_count_gensim',
    9: 'lsi_tfidf_gensim',
}

code = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
}

if __name__ == '__main__':
    file_path = sys.argv[1] # path and filename
    no_tests =  int(sys.argv[2]) # number of tests

    for i in range(0, 10):
        code[i].append('ArrayList<Double> ARI_sum_' + file_path + d[i] + ' = new ArrayList<Double>();')
        code[i].append('ArrayList<Double> PUR_sum_' + file_path + d[i] + ' = new ArrayList<Double>();')
        code[i].append('ArrayList<Double> time_sum_' + file_path + d[i] + ' = new ArrayList<Double>();')
        
    
    for i in range(1, no_tests + 1):
        file_name = file_path + str(i)

        with codecs.open(file_name, 'r',  "utf-8") as current_file:
            idx1 = 0
            idx2 = 0
            for line in current_file:
                if line.startswith("{"):                    
                    if idx1 == 10:
                        idx1 = 0
                    elem = ast.literal_eval(line)
                    mat, a, b = calMatrices(elem)
                    eval_var = file_path + d[idx1] + '_' + str(i)
                    ari_var = 'ARI_sum_' + file_path + d[idx1]
                    pur_var = 'PUR_sum_' + file_path + d[idx1]
                    try:
                        code[idx1].append( 'Double[][] ' + eval_var + ' = ' + printMatrix(mat) + ';' )
                        code[idx1].append( ari_var + '.add(evalARI(' + eval_var + '));' )
                        code[idx1].append( pur_var + '.add(evalPurity(' + eval_var + '));' )
                    except:
                        print current_file
                    idx1 += 1
                if idx2 == 10:
                    idx2 = 0               
                if alg[idx2] in line:
                    time_var = 'time_sum_' + file_path + d[idx2]
                    code[idx2].append( time_var + '.add(' + line[line.index('time')+5:].replace('\n', '').replace('\r', '') + ');' )
                    idx2 += 1
    
    for i in range(0, 10):
        code[i].append( 'System.out.println(\"ARI '+d[i] + '\");' )
        code[i].append( 'Collections.sort( ARI_sum_' + file_path + d[i] + ');' )
        code[i].append( 'stats(ARI_sum_' + file_path + d[i] + ');' )
        code[i].append( 'System.out.println(\"PUR '+d[i] + '\");' )
        code[i].append( 'Collections.sort( PUR_sum_' + file_path + d[i] + ');' )
        code[i].append( 'stats(PUR_sum_' + file_path + d[i] + ');' )
        code[i].append( 'System.out.println(\"Time '+d[i] + '\");' )
        code[i].append( 'Collections.sort( time_sum_' + file_path + d[i] + ');' )
        code[i].append( 'stats(time_sum_' + file_path + d[i] + ');' )

    for i in range(0, 10):
        for elem in code[i]:
            print elem
        print '\n\n'