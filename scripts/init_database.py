from manufacturing_ai_copilot.db.schema import initialize_database
from manufacturing_ai_copilot.db.seed import seed_work_orders


def main() -> None:
    # 先创建表结构，再写入可重复执行的种子数据。
    initialize_database()

    inserted_count = seed_work_orders()

    print(f"Database initialized: inserted={inserted_count}")


if __name__ == "__main__":
    main()
