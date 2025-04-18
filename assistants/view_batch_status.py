
from batch_infra import check_batch


def main():
    import sys
    batch_status = check_batch(sys.argv[1])
    print(batch_status)

if __name__ == "__main__":
    main()
