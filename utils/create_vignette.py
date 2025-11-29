import sys
from src.llm_system_prompt import field_definitions
from utils.utils import row_to_json
import pandas as pd

patient_data = {
  "date_crf": "2024-02-13",
  "unit_crf": 3,
  "age_crf": 60,
  "sex_crf": 1,
  "occupation1_crf": 1,
  "occupation2_crf": 14,
  "education_crf": 5,
  "income_crf": 2,
  "family_crf": 2,
  "marital": 2,
  "polygamy": 0,
  "pregnancy_crf": None,
  "comorbidity___1": 0,
  "comorbidity___2": 1,
  "comorbidity___3": 0,
  "comorbidity___4": 0,
  "comorbidity___5": 0,
  "comorbidity___6": 0,
  "comorbidity___7": 0,
  "comorbidity___8": 0,
  "comorbidity___9": 0,
  "comorbidity___11": 0,
  "comorbidity___12": 0,
  "comorbidity___13": 0,
  "comorbidity___14": 0,
  "comorbidity___15": 0,
  "comorbidity___17": 1,
  "comorbidity___18": 0,
  "state_crf": 30,
  "community_crf": 2,
  "ethnicity_crf": 9,
  "temperature": 41.2,
  "sbp": 160,
  "dbp": 110,
  "pulse_rate": 115,
  "weight": 84,
  "respiratory_rate": 30,
  "fever___1": 1,
  "lethargy___1": 0,
  "headache___1": 1,
  "vision___1": 0,
  "cough___1": 1,
  "joint___1": 1,
  "muscle___1": 1,
  "dyspnoea___1": 1,
  "wheezing___1": 0,
  "ear___1": 0,
  "appetite___1": 1,
  "chest___1": 0,
  "swallowing___1": 0,
  "nausea___1": 0,
  "vomit___1": 0,
  "diarrhea___1": 0,
  "abdominal___1": 1,
  "back___1": 1,
  "hiccups___1": 1,
  "mouth___1": 0,
  "throat___1": 0,
  "nose___1": 0,
  "rash___1": 0,
  "seizure___1": 0,
  "bruising___1": 0,
  "confusion___1": 0,
  "swelling___1": 0,
  "neck___1": 0,
  "urine___1": 1,
  "stoolblood___1": 0,
  "noseblood___1": 0,
  "oralblood___1": 0,
  "vomitblood___1": 0,
  "vagblood___1": 0,
  "intravenousblood___1": 0,
  "othersymptoms___1": 0,
  "feverd": 2,
  "headached": 2,
  "cough_details___1": 1,
  "cough_details___2": 0,
  "cough_details___3": 0,
  "symptomhistory": 0,
  "housingtype___1": 1,
  "housingtype___2": 0,
  "housingtype___3": 0,
  "housingtype___4": 0,
  "roofingtype___0": 0,
  "roofingtype___1": 1,
  "roofingtype___2": 0,
  "roofingtype___3": 0,
  "roofingtype___4": 0,
  "housingrisk___1": 0,
  "housingrisk___2": 0,
  "housingrisk___3": 0,
  "housingrisk___4": 0,
  "housingrisk___5": 0,
  "housingrisk___6": 0,
  "housingrisk___7": 1,
  "housingrisk___8": 0,
  "mosquitonet": 1,
  "mosquitobite": 1,
  "insecticide": 1,
  "rodent_contact": 1,
  "rodent_touch": 0,
  "domestic_contact": 1,
  "wild_contact": 0,
  "uncooked_meat": 0,
  "food_sacs": 1,
  "food_bowls": 0,
  "food_container": 1,
  "eat_rodent": 0,
  "eat_bush": 1,
  "unprotected_sex": 1,
  "sharp_objects": 0,
  "blood_transfusion": 0,
  "haircut": 1,
  "toilet": 1,
  "surgery": 0,
  "other_contacts": 0,
  "family_sick": 1,
  "family_care": 0,
  "family_room": 0,
  "family_bed": 0,
  "family_fluids": 0,
  "hc": 0,
  "hc_ppe": None,
  "contact_fluids": None,
  "travel": 1,
  "travel_contact": 0,
  "funeral": 0,
  "funeral_body": None,
  "clean_water": 1,
  "watersource___1": 0,
  "watersource___2": 0,
  "watersource___3": 0,
  "watersource___4": 1,
  "watersource___5": 1,
  "watersource___6": 1,
  "watersource___7": 1,
  "watersource___8": 1,
  "watersource___9": 0,
  "color_urin": 1,
  "appearance_urin": 1,
  "leukocyte_urin": 1,
  "nitrite_urin": 1,
  "protein_urin": 0,
  "blood_urin": 3,
  "ketones_urin": 0,
  "bilirubin_urin": 1,
  "glucose_urin": 1
}

patient_data_2 = {'date_crf': '2024-01-16', 'unit_crf': 1, 'age_crf': 25, 'sex_crf': 2, 'occupation1_crf': 4, 'occupation2_crf': None, 'education_crf': 4, 'income_crf': 3, 'family_crf': 1, 'marital': 2, 'polygamy': 0, 'pregnancy_crf': 1, 'comorbidity___1': 0, 'comorbidity___2': 0, 'comorbidity___3': 0, 'comorbidity___4': 0, 'comorbidity___5': 0, 'comorbidity___6': 0, 'comorbidity___7': 0, 'comorbidity___8': 0, 'comorbidity___9': 0, 'comorbidity___11': 0, 'comorbidity___12': 0, 'comorbidity___13': 0, 'comorbidity___14': 0, 'comorbidity___15': 0, 'comorbidity___17': 0, 'comorbidity___18': 1, 'state_crf': 29, 'community_crf': 2, 'ethnicity_crf': 9, 'temperature': 36.6, 'sbp': 115, 'dbp': 60, 'pulse_rate': 86, 'weight': 66, 'respiratory_rate': 21.79704589017928, 'fever___1': 0, 'lethargy___1': 0, 'headache___1': 1, 'vision___1': 0, 'cough___1': 1, 'joint___1': 0, 'muscle___1': 0, 'dyspnoea___1': 0, 'wheezing___1': 0, 'ear___1': 0, 'appetite___1': 1, 'chest___1': 0, 'swallowing___1': 0, 'nausea___1': 1, 'vomit___1': 0, 'diarrhea___1': 0, 'abdominal___1': 1, 'back___1': 1, 'hiccups___1': 0, 'mouth___1': 0, 'throat___1': 0, 'nose___1': 0, 'rash___1': 0, 'seizure___1': 0, 'bruising___1': 0, 'confusion___1': 0, 'swelling___1': 0, 'neck___1': 0, 'urine___1': 0, 'stoolblood___1': 0, 'noseblood___1': 0, 'oralblood___1': 0, 'vomitblood___1': 0, 'vagblood___1': 0, 'intravenousblood___1': 0, 'othersymptoms___1': 1, 'feverd': None, 'headached': 2, 'cough_details___1': 0, 'cough_details___2': 0, 'cough_details___3': 1, 'symptomhistory': 0, 'housingtype___1': 1, 'housingtype___2': 0, 'housingtype___3': 0, 'housingtype___4': 0, 'roofingtype___0': 0, 'roofingtype___1': 1, 'roofingtype___2': 0, 'roofingtype___3': 0, 'roofingtype___4': 0, 'housingrisk___1': 0, 'housingrisk___2': 0, 'housingrisk___3': 0, 'housingrisk___4': 0, 'housingrisk___5': 0, 'housingrisk___6': 0, 'housingrisk___7': 1, 'housingrisk___8': 0, 'mosquitonet': 0, 'mosquitobite': 0, 'insecticide': 0, 'rodent_contact': 0, 'rodent_touch': None, 'domestic_contact': 0, 'wild_contact': 0, 'uncooked_meat': 1, 'food_sacs': 1, 'food_bowls': 0, 'food_container': 1, 'eat_rodent': 0, 'eat_bush': 0, 'unprotected_sex': 1, 'sharp_objects': 0, 'blood_transfusion': 0, 'haircut': 0, 'toilet': 0, 'surgery': 0, 'other_contacts': 0, 'family_sick': 0, 'family_care': None, 'family_room': None, 'family_bed': None, 'family_fluids': None, 'hc': 0, 'hc_ppe': None, 'contact_fluids': None, 'travel': 0, 'travel_contact': None, 'funeral': 0, 'funeral_body': None, 'clean_water': 1, 'watersource___1': 0, 'watersource___2': 0, 'watersource___3': 0, 'watersource___4': 0, 'watersource___5': 0, 'watersource___6': 0, 'watersource___7': 1, 'watersource___8': 1, 'watersource___9': 0, 'color_urin': 0, 'appearance_urin': 1, 'leukocyte_urin': 2, 'nitrite_urin': 0, 'protein_urin': 1, 'blood_urin': 0, 'ketones_urin': 0, 'bilirubin_urin': 0, 'glucose_urin': 0, 'record_id': '1-02-24-1645'}
df_field_definitions = pd.DataFrame(eval(field_definitions))

def find_keys_to_scan(data):
    keys_to_scan = []
    for key in data:
        if '___' in key:
            keys_to_scan.append(key.split('___')[0])
        else:
            keys_to_scan.append(key)
    return set(keys_to_scan)

def gather_data_specific_key(data, query_key):
    results = []
    for key in data:
        if query_key in key:
            if '___' in key:
                value = key.split('___')[1]
                if data[key] == 1:
                    results.append(value)
            elif query_key == key:
                results.append(data[key])
    return results

def generate_patient_data_vignette(patient_data):
    
    useful_mappings = {
        'Indicate your housing material': 'housing material',
        'feverd': 'fever duration',
        'headached': 'headache duration',
        'Indicate your roofing type': 'roofing type',
        'Is your housing located around any of the following:Check as applicable': 'house is located near',
        'Any known comorbidities and/or chronic infections?': 'has a comorbidity/known chronic infection of',
        'Which of the sources of water do you use?Check as applicable': 'water source used',
        'Appearance': 'appearance of urine',
        'Nitrite': 'nitrite in urine',
        'joint': 'joint pain',
        'abdominal': 'abdominal pain',
        'Back': 'back pain',
        'muscle': 'muscle pain',
    }

    patient_vignette = ''
    for i in find_keys_to_scan(patient_data):
        try:
            options = df_field_definitions[df_field_definitions["Field Name"] == i].get("Choices", None)
            field_label = df_field_definitions[df_field_definitions["Field Name"] == i].get("Field Label", None).item()
            if options.isna().item() is not True:
                options = options.item().split("|")
                options = {int(x.split(",")[0].strip()): x.split(",")[1].strip() for x in options}
                pat_dat = gather_data_specific_key(patient_data, i)
                for res in pat_dat:
                    if res is not None and int(res) in options:
                        field_label = useful_mappings.get(field_label, field_label)
                        if options[int(res)] == '':
                            patient_vignette += f'The patients has {field_label}.'
                        else:
                            patient_vignette += f'The patients {field_label} is {options[int(res)]}. '
            else:
                pat_dat = gather_data_specific_key(patient_data, i)
                if i == 'toilet' and 1 in pat_dat:
                    patient_vignette += 'The patients has used a public toilet. '
                elif i == 'weight':
                    patient_vignette += f'The patients weight is {pat_dat[0]}. '
                elif i == 'pulse_rate':
                    patient_vignette += f'The patients pulse rate is {pat_dat[0]}. '
                elif i == 'family_bed' and 1 in pat_dat:
                    patient_vignette += 'The patients was sleeping in a family bed. '
                elif i == 'bush meet' and 1 in pat_dat:
                    patient_vignette += 'The patients has recently eaten bush meat. '
                elif i == 'wild_contact' and 1 in pat_dat:
                    patient_vignette += 'The patients has had contact with wild animals. '
                elif i == 'unprotected_sex' and 1 in pat_dat:
                    patient_vignette += 'The patients has had unprotected sex. '
                elif i == 'domestic_contact' and 1 in pat_dat:
                    patient_vignette += 'The patients has had contact with a domestic animal. '
                elif i == 'rodent_touch' and 1 in pat_dat:
                    patient_vignette += 'The patients has touched a rodent. '
                elif i == 'sharp_objects' and 1 in pat_dat:
                    patient_vignette += 'The patients has had contact with sharp objects. '
                elif i == 'food_bowls' and 1 in pat_dat:
                    patient_vignette += 'The patients stores food in open bowls. '
                elif i == 'other_contacts' and 1 in pat_dat:
                    patient_vignette += 'The patients has had other contacts. '
                elif i == 'sbp':
                    patient_vignette += f'The patients systolic blood pressure is {pat_dat[0]}. '
                elif i == 'family_room' and 1 in pat_dat:
                    patient_vignette += 'The patients was sleeping in the same room as their family. '
                elif i == 'travel' and 1 in pat_dat:
                    patient_vignette += 'The patients has traveled recently. '
                elif i == 'funeral_body' and 1 in pat_dat:
                    patient_vignette += 'The patients has handled a dead body. '
                elif i == 'insecticides' and 1 in pat_dat:
                    patient_vignette += 'The patients uses insecticides. '
                elif i == 'clean_water' and 1 in pat_dat:
                    patient_vignette += 'The patients has access to clean water. '
                elif i == 'travel_contact' and 1 in pat_dat:
                    patient_vignette += 'The patient has been in a vehicle with a sick person. '
                elif i == 'blood_transfusion' and 1 in pat_dat:
                    patient_vignette += 'The patients has recently received a blood transfusion. '
                elif i == 'family_care' and 1 in pat_dat:
                    patient_vignette += 'The patients has cared for a family member. '
                elif i == 'uncooked_meat' and 1 in pat_dat:
                    patient_vignette += 'The patients has touched uncooked meat. '
                elif i == 'food_container' and 1 in pat_dat:
                    patient_vignette += 'The patients stores food in covered containers. '
                elif i == 'eat_rodent' and 1 in pat_dat:
                    patient_vignette += 'The patients has eaten rodents. '
                elif i == 'mosquito_bite' and 1 in pat_dat:
                    patient_vignette += 'The patients has bite marks of mosquitos or other insects. '
                elif i == 'food_sacs' and 1 in pat_dat:
                    patient_vignette += 'The patients stores food in food sacs. '
                elif i == 'family_fluids' and 1 in pat_dat:
                    patient_vignette += 'The patients has been exposed to family fluids. '
                elif i == 'symptomhistory' and 1 in pat_dat:
                    patient_vignette += 'The patients has a history of similar symptoms. '
                elif i == 'temperature':
                    patient_vignette += f'The patients temperature is {pat_dat[0]}. '
                elif i == 'mosquitonet' and 1 in pat_dat:
                    patient_vignette += 'The patients uses a mosquito net. '
                elif i == 'dbp':
                    patient_vignette += f'The patients diastolic blood pressure is {pat_dat[0]}. '
                elif i == 'hc_ppe' and 1 in pat_dat:
                    patient_vignette += 'The patients uses personal protective equipment. '
                elif i == 'respiratory_rate':
                    patient_vignette += f'The patients respiratory rate is {pat_dat[0]}. '
                elif i == 'funeral' and 1 in pat_dat:
                    patient_vignette += 'The patients has attended a funeral. '
                elif i == 'family_sick' and 1 in pat_dat:
                    patient_vignette += 'The patient has had recent sick family members. '
                elif i == 'contact_fluids' and 1 in pat_dat:
                    patient_vignette += 'The patient has been in contact with blood or body fluids. '
                elif i == 'haircut' and 1 in pat_dat:
                    patient_vignette += 'The patient has had a recent haircut with a public or shared clipper. '
                elif i == 'hc' and 1 in pat_dat:
                    patient_vignette += 'The patients has attended to a sick person as a healthcare worker. '
                elif i == 'surgery' and 1 in pat_dat:
                    patient_vignette += 'The patients has undergone surgery. '
        except Exception as e:
            print(f'Errored here with this error {e} for this {i}')
            import pdb; pdb.set_trace()
    
    return patient_vignette
