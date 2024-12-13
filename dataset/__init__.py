from dataset.vl_cmu_cd import get_VL_CMU_CD
from dataset.pcd import get_GSV, get_TSUNAMI
from dataset.changesim import get_ChangeSim_Multi

dataset_dict = {
    'ChangeSim_Multi': get_ChangeSim_Multi,
    "VL_CMU_CD": get_VL_CMU_CD,
    'GSV': get_GSV,
    'TSUNAMI': get_TSUNAMI,
}
