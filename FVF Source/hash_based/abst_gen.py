from utils import rdb
from tqdm import tqdm
import hashlib
from simhash import Simhash
from abstractions import abst_method1, abst_method2, abst_method3

db = rdb.DBConnection()


def do_work1(project_id, func_id, func, lang):
    abst1 = abst_method1(func.encode(errors="ignore"), lang)
    abst1_join = b"".join(abst1)

    db.execute(
        """
        UPDATE vuln_func SET abst1 = %s WHERE func_id = %s
    """,
        (abst1_join, func_id),
    )

    # sha1
    abst1_sha1 = hashlib.sha1(abst1_join).hexdigest()
    db.execute(
        """
        UPDATE vuln_func SET abst1_sha1 = %s WHERE func_id = %s
    """,
        (abst1_sha1, func_id),
    )

    # lsh, sliding window of 3
    abst1_ft = [
        (abst1[i] + abst1[i + 1] + abst1[i + 2]).decode()
        for i in range(0, len(abst1) - 2)
    ]
    abst1_lsh = Simhash(abst1_ft, f=64).value
    abst1_lsh = str(abst1_lsh)
    db.execute(
        """
        UPDATE vuln_func SET abst1_lsh = %s WHERE func_id = %s
    """,
        (abst1_lsh, func_id),
    )


def do_work2(project_id, func_id, func, lang):
    abst2 = abst_method2(func.encode(errors="ignore"), lang)
    abst2_join = b"".join(abst2)

    db.execute(
        """
        UPDATE vuln_func SET abst2 = %s WHERE func_id = %s
    """,
        (abst2_join, func_id),
    )

    # sha1
    abst2_sha1 = hashlib.sha1(abst2_join).hexdigest()
    db.execute(
        """
        UPDATE vuln_func SET abst2_sha1 = %s WHERE func_id = %s
    """,
        (abst2_sha1, func_id),
    )

    # lsh, sliding window of 3
    abst2_ft = [
        (abst2[i] + abst2[i + 1] + abst2[i + 2]).decode()
        for i in range(0, len(abst2) - 2)
    ]
    abst2_lsh = Simhash(abst2_ft, f=64).value
    abst2_lsh = str(abst2_lsh)
    db.execute(
        """
        UPDATE vuln_func SET abst2_lsh = %s WHERE func_id = %s
    """,
        (abst2_lsh, func_id),
    )


def do_work3(project_id, func_id, func, lang):
    abst3 = abst_method3(func.encode(errors="ignore"), lang)
    abst3_join = b"".join(abst3)

    db.execute(
        """
        UPDATE vuln_func SET abst3 = %s WHERE func_id = %s
    """,
        (abst3_join, func_id),
    )

    # sha1
    abst3_sha1 = hashlib.sha1(abst3_join).hexdigest()
    db.execute(
        """
        UPDATE vuln_func SET abst3_sha1 = %s WHERE func_id = %s
    """,
        (abst3_sha1, func_id),
    )

    # lsh, sliding window of 3
    abst3_ft = [
        (abst3[i] + abst3[i + 1] + abst3[i + 2]).decode()
        for i in range(0, len(abst3) - 2)
    ]
    abst3_lsh = Simhash(abst3_ft, f=64).value
    abst3_lsh = str(abst3_lsh)
    db.execute(
        """
        UPDATE vuln_func SET abst3_lsh = %s WHERE func_id = %s
    """,
        (abst3_lsh, func_id),
    )


# total = db.execute("SELECT COUNT(1) FROM vuln_func")[0][0]
dataset = db.execute(
    "SELECT project_id, func_id, func, file_lang FROM vuln_func WHERE abst1 IS NULL"
)

for project_id, func_id, func, lang in tqdm(dataset):
    do_work1(project_id, func_id, func, lang)
    do_work2(project_id, func_id, func, lang)
    do_work3(project_id, func_id, func, lang)
