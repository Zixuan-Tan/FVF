from utils import codeparser


def abst_method1(func, lang):
    # target: find semantic equivalent functions
    return codeparser.abstract_func_clike(
        func,
        lang,
        abstract_fname=codeparser.ABST_AS_TYPE,
        abstract_lvar=codeparser.ABST_WITH_NUM,
        abstract_fparam=codeparser.ABST_WITH_NUM,
        abstract_label=codeparser.ABST_WITH_NUM,
        abstract_gsym=codeparser.ABST_WITH_NUM,
        abstract_field=False,
        abstract_type=codeparser.ABST_AS_TYPE,
        abstract_literal=codeparser.ABST_AS_TYPE,
        abstract_func_call=codeparser.ABST_AS_TYPE | codeparser.ABST_NON_SYS,
    )


def abst_method2(func, lang):
    # target: find fully equivalent functions
    return codeparser.abstract_func_clike(
        func,
        lang,
        abstract_fname=codeparser.ABST_AS_TYPE,
        abstract_lvar=False,
        abstract_fparam=False,
        abstract_label=False,
        abstract_gsym=False,
        abstract_field=False,
        abstract_type=False,
        abstract_literal=False,
        abstract_func_call=False,
    )


def abst_method3(func, lang):
    # target: hw type2
    return codeparser.abstract_func_clike(
        func,
        lang,
        abstract_fname=codeparser.ABST_AS_TYPE,
        abstract_lvar=False,
        abstract_fparam=codeparser.ABST_AS_TYPE,
        abstract_label=False,
        abstract_gsym=False,
        abstract_field=False,
        abstract_type=False,
        abstract_literal=False,
        abstract_func_call=False,
    )
