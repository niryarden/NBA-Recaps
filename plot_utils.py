import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


def plot_stats(results):
    print(results.describe())


def correlation_analysis(results):
    numeric_df = results.drop(columns=['game_id'], errors='ignore').select_dtypes(include='number')

    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt='.2f', annot_kws={"size": 10})

    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()


def distribution_plots(results):
    results.hist(bins=20, figsize=(10, 6))
    plt.suptitle('Distribution of Metrics')
    plt.show()


def box_plot(results):
    results.plot.box()
    plt.title('Box Plot of Metric Scores')
    plt.show()


def top_bottom_games(results, metric, n=5):
    top_games = results.nlargest(n, metric)
    bottom_games = results.nsmallest(n, metric)

    print(f"Top {n} games based on {metric}:")
    print(top_games[['game_id', metric]])

    print(f"\nBottom {n} games based on {metric}:")
    print(bottom_games[['game_id', metric]])

    return top_games, bottom_games


def scatter_plot(results, metric_x, metric_y):
    plt.scatter(results[metric_x], results[metric_y])
    plt.xlabel(metric_x)
    plt.ylabel(metric_y)
    plt.title(f'Scatter Plot of {metric_x} vs {metric_y}')
    plt.show()


def trend_analysis(results_df):
    results_df = results_df.reset_index(drop=True)  # Reset index to numerical

    results_df[['nba_overlap_score', 'player_names_overlap_score', 'sbert_similarity_score']].plot()

    plt.xlabel('Game Number')
    plt.ylabel('Score')
    plt.title('Metric Scores Over Games')
    plt.show()


def clustering_analysis(results_df, n_clusters=3):
    numeric_df = results_df.drop(columns=['game_id'], errors='ignore').select_dtypes(include='number')
    kmeans = KMeans(n_clusters=n_clusters)
    numeric_df['cluster'] = kmeans.fit_predict(numeric_df)

    sns.scatterplot(data=numeric_df, x='nba_overlap_score', y='sbert_similarity_score', hue='cluster', palette='Set1')
    plt.title('Clustering of Games Based on Metrics')
    plt.show()


def scatter_plot_3d(results_df):
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3 metrics in 3D space
    x = results_df['nba_overlap_score']
    y = results_df['player_names_overlap_score']
    z = results_df['sbert_similarity_score']

    scatter = ax.scatter(x, y, z, c='b', marker='o')

    # Set axis labels
    ax.set_xlabel('NBA Overlap Score')
    ax.set_ylabel('Player Names Overlap Score')
    ax.set_zlabel('SBERT Similarity Score')

    plt.show()


def analyze_metrics(results_df):
    # 1. Descriptive Statistics
    print("Descriptive Statistics:")
    plot_stats(results_df)
    print("\n")

    # 2. Correlation Analysis
    print("Correlation Analysis:")
    correlation_analysis(results_df)
    print("\n")

    # 3. Distribution Plots
    print("Distribution Plots:")
    distribution_plots(results_df)
    print("\n")

    # 4. Box Plot
    print("Box Plot:")
    box_plot(results_df)
    print("\n")

    # 5. Top and Bottom Games based on each metric
    print("Top and Bottom Games for NBA Overlap Score:")
    top_bottom_games(results_df, 'nba_overlap_score')
    print("\n")

    print("Top and Bottom Games for Player Names Overlap Score:")
    top_bottom_games(results_df, 'player_names_overlap_score')
    print("\n")

    print("Top and Bottom Games for SBERT Similarity Score:")
    top_bottom_games(results_df, 'sbert_similarity_score')
    print("\n")

    # 6. Scatter Plot (between NBA Overlap and SBERT Similarity)
    print("Scatter Plot between NBA Overlap Score and SBERT Similarity Score:")
    scatter_plot(results_df, 'nba_overlap_score', 'sbert_similarity_score')
    print("\n")

    # 6. Scatter Plot (between NBA Overlap and SBERT Similarity)
    print("Scatter Plot between NBA Overlap Score and SBERT Similarity Score:")
    scatter_plot(results_df, 'nba_overlap_score', 'player_names_overlap_score')
    print("\n")

    # 6. Scatter Plot (between NBA Overlap and SBERT Similarity)
    print("Scatter Plot between NBA Overlap Score and SBERT Similarity Score:")
    scatter_plot(results_df, 'player_names_overlap_score', 'sbert_similarity_score')
    print("\n")

    # 7. Trend Analysis over Games
    print("Trend Analysis of Metric Scores over Games:")
    trend_analysis(results_df)
    print("\n")

    # 8. Clustering Analysis (3 clusters by default)
    print("Clustering Analysis:")
    clustering_analysis(results_df)
    print("\n")

    # 9. Scatter Plot 3D
    print("Scatter Plot 3D:")
    scatter_plot_3d(results_df)
    print("\n")
