import json
import matplotlib.pyplot as plt


def split_into_buckets(lst, maximum):
    bucket_size = 100
    bucket_dict = {}
    for i in range(bucket_size, maximum + 2 * bucket_size, bucket_size):
        bucket_dict[i] = 0

    for num in lst:
        bucket = (int(num / bucket_size) + 1) * bucket_size
        bucket_dict[bucket] += 1

    return bucket_dict


def graph():
    with open("/cs/labs/roys/nir.yarden/anlp-project/NBA-Recaps/graphs/length.json", 'r') as f:
        lengths = json.loads(f.read())

    print(sum(lengths) / len(lengths))
    maximum = (int(max(lengths) / 100)) * 100
    bucket_dict = split_into_buckets(lengths, maximum)
    x = list(bucket_dict.keys())
    y = list(bucket_dict.values())

    plt.bar(x[36:70], y[36:70], width=80)
    plt.xlabel('Processed play-by-play length (Tokens)')
    plt.ylabel('Samples')
    plt.title('Processed play-by-play length distribution')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig('/cs/labs/roys/nir.yarden/anlp-project/NBA-Recaps/graphs/processed_input_length.png')


def precentage():
    with open("/cs/labs/roys/nir.yarden/anlp-project/NBA-Recaps/graphs/length.json", 'r') as f:
        lengths = json.loads(f.read())
    for threashold in [5500, 5800, 6000, 6200, 6500]:
        small_enough = 0
        for length in lengths:
            if length < threashold:
                small_enough += 1
        print(f"{100 * small_enough / len(lengths)}% above {threashold}")


if __name__ == "__main__":
    graph()
    precentage()
