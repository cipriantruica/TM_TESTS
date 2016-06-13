package main.mallet.news;

import java.util.*;

import evaluation.ClusterEvaluator;
import evaluation.ContingencyTable;
import main.mallet.news.tests.*;

public class Main_news_lda{

	public static double evalPurity(Double[][] confusionMatrix){
		ContingencyTable table = new ContingencyTable(transposeMatrix(confusionMatrix));
		ClusterEvaluator ev = new ClusterEvaluator();
		ev.setData(table);
		return ev.getPurity();
	}

	public static double evalARI(Double[][] confusionMatrix){
		ContingencyTable table = new ContingencyTable(transposeMatrix(confusionMatrix));
		ClusterEvaluator ev = new ClusterEvaluator();
		ev.setData(table);	
		return ev.getAdjustedRandIndex();
	}
	
	public static void stats(ArrayList<Double> myList){
		double average = 0.0;
		double std = 0.0;

		for (double elem : myList){
			average += elem;
		}
		average /= myList.size();
		
		for (double elem : myList){
			std += Math.pow(elem - average, 2);
		}
		std /= myList.size();
		std = Math.sqrt(std);
		
		System.out.printf("%f\t%f\n", average, std);
	}
	
	public static Double[][] transposeMatrix(Double[][] mat){
		Double[][] trans_mat = new Double[mat[0].length][mat.length];
		for (int i = 0; i < mat.length; i++){
            for (int j = 0; j < mat[0].length; j++){
                trans_mat[j][i] = mat[i][j];
            }
		}
		return trans_mat;
		
	}
		
	
	public static void main(String[] args){
		ArrayList<Double> ARI_sum_news_cleantText = new ArrayList<Double>();
		ArrayList<Double> PUR_sum_news_cleantText = new ArrayList<Double>();
		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_1().news_lda_cleanText1));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_1().news_lda_cleanText1));

		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_1().news_lda_cleanText2));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_1().news_lda_cleanText2));

		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_1().news_lda_cleanText3));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_1().news_lda_cleanText3));

		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_1().news_lda_cleanText4));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_1().news_lda_cleanText4));

		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_1().news_lda_cleanText5));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_1().news_lda_cleanText5));

		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_1().news_lda_cleanText6));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_1().news_lda_cleanText6));

		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_1().news_lda_cleanText7));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_1().news_lda_cleanText7));

		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_1().news_lda_cleanText8));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_1().news_lda_cleanText8));

		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_1().news_lda_cleanText9));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_1().news_lda_cleanText9));

		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_1().news_lda_cleanText10));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_1().news_lda_cleanText10));
		
		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_2().news_lda_cleanText11));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_2().news_lda_cleanText11));
		
		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_2().news_lda_cleanText12));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_2().news_lda_cleanText12));
		
		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_2().news_lda_cleanText13));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_2().news_lda_cleanText13));
		
		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_2().news_lda_cleanText14));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_2().news_lda_cleanText14));
		
		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_2().news_lda_cleanText15));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_2().news_lda_cleanText15));
		
		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_2().news_lda_cleanText16));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_2().news_lda_cleanText16));
		
		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_2().news_lda_cleanText17));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_2().news_lda_cleanText17));
		
		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_2().news_lda_cleanText18));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_2().news_lda_cleanText18));
		
		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_2().news_lda_cleanText19));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_2().news_lda_cleanText19));
		
		ARI_sum_news_cleantText.add(evalARI(new News_CleanText_2().news_lda_cleanText20));
		PUR_sum_news_cleantText.add(evalPurity(new News_CleanText_2().news_lda_cleanText20));
		
		System.out.println("ARI Clean Text");
		Collections.sort( ARI_sum_news_cleantText);
		stats(ARI_sum_news_cleantText);
		System.out.println("PUR Clean Text");
		Collections.sort( PUR_sum_news_cleantText);
		stats(PUR_sum_news_cleantText);

		
	}
	
}
