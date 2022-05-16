import csv
import re

def csvopen(file):
    with open(file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        data = {'Food_Name':[],'Energi':[], 'Natrium':[], 'Jumlah Sajian Per Kemasan':[], 'Jumlah Per Sajian':[], 'Protein':[], 'Karbohidrat total':[], 'Takaran Saji':[], 'Lemak total':[], 'Air':[], 'Abu':[], 'Karoten':[], 'Karoten total':[]}
        label = ['Energi', 'Natrium', 'Jumlah Sajian Per Kemasan', 'Jumlah Per Sajian', 'Protein', 'Karbohidrat total', 'Takaran Saji', 'Lemak total', 'Air', 'Abu', 'Karoten', 'Karoten total']
        for index,row in enumerate(reader):
            name = row['food_name']
            data['Food_Name'].append(name)
            infos = row['food_info'].split(', ')
            val_label = []
            for info in infos:
                label_info = re.findall('([a-zA-Z ]*) (.*)', info)[0][0]
                value = re.findall('([a-zA-Z ]*) (.*)', info)[0][1]
                data[label_info].append(value)
                val_label.append(label_info)
        
            for i in label:
                if i not in val_label:
                    data[i].append('N/A')
        
        return data, label

def csvwrite(file, data, label):
    with open(file, 'w', newline='') as csvfile:
        fieldnames = ['Food_Name']
        for i in label:
            fieldnames.append(i)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,delimiter=';')

        writer.writeheader()
        for index,_ in enumerate(data['Food_Name']):
            writer.writerow({'Food_Name': data['Food_Name'][index],
                fieldnames[1]: data[fieldnames[1]][index],
                fieldnames[2]: data[fieldnames[2]][index],
                fieldnames[3]: data[fieldnames[3]][index],
                fieldnames[4]: data[fieldnames[4]][index],
                fieldnames[5]: data[fieldnames[5]][index],
                fieldnames[6]: data[fieldnames[6]][index],
                fieldnames[7]: data[fieldnames[7]][index],
                fieldnames[8]: data[fieldnames[8]][index],
                fieldnames[9]: data[fieldnames[9]][index],
                fieldnames[10]: data[fieldnames[10]][index],
                fieldnames[11]: data[fieldnames[11]][index],
                fieldnames[12]: data[fieldnames[12]][index]
                })

if __name__ == '__main__':
    data, label = csvopen('data.csv')

    csvwrite('new_data.csv', data, label)
        

