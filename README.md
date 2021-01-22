
# Git Nedir?
Kısaca **Git** bir versiyon kontrol sistemidir diyebiliriz. Bu **açık kaynaklı** versiyon kontrol sistemi sayesinde; projelerimizin versiyon kontrol takibini yapabilir, dünya çapında projelere katkıda bulunabilir, kendi projelerimizi dünya ile paylaşabilir ve projelerimizin mobilitesini artırabiliriz. Bu sistem sayesinde, bir projede aynı anda birden fazla kişi ile çalışabilir ve bir dosyada yapılan tüm değişiklikleri görüntüleyebiliriz.
**Git != GitHub || GitLab**

# Temel Git Terimleri
- **Repository**
  - Bir diğer adıyla **Kod Deposu**. Proje dizinimizde  ``` git init``` diyerek bir repository oluşturabiliriz.
- **Working Directory**
  - Kısaca **Çalışma Alanı**. Projemizin ana dizinidir. Bir dosyanın/klasörün **Working Directory**'de bulunması o dosyanın **Git** tarafından takip edildiği anlamına gelmez. Sadece **Staging Area**'daki dosyalar **Git** tarafından takip edilir.
- **Staging Area**
  - **Git** tarafından **takip edilen** dosyalar burada bulunur. Gerçek manada **fiziki bir alan değildir**. Dosyalarımızın durumunu belirten hayali bir ortam olarak düşünebiliriz. 
- **Commit**
  - Yaptığımız değişikliklere verdiğimiz kısa açıklamalardır. ilgili dosya/dizinde yaptığımız değişiklikler **repository**'mize kaydedilir.

# Kurulum
[Git](https://git-scm.com/)'in official web sitesi üzerinden kurulum dosyasını indirebiliriz. Eğer ki Linux kullanıcısı isek (Debian tabanlı dağıtımlar için) `apt-get install git` diyerek sistemimize Git'i kurabiliriz.
- Git versiyon kontrol sisteminin yine Git ile kontrol edildiğini de belirtmekte fayda var.


# Temel Terminal/Komut İstemcisi Bilgisi
- **Terminal Nasıl Açılır?**
  - **Linux** Dağıtımlar:
    - **CTRL + ALT + T** tuş kombinasyonu ile Terminal açılabilir (Gnome masaüstünü kullananlar için). Uygulamalar penceresinden de Terminal'i seçerek açabilirsiniz.
  - **OS X**:
    - **cmd + space(boşluk)** tuş kombinasyonu ile Spotlight'ı açtıktan sonra **Terminal** yazıp Enter'a basarak açabilirsiniz.
  - **Windows**:
    - **ALT + R** tuş kombinasyonu sayesinde açılan pencerede **cmd** yazarak erişebilirsiniz.
- **Terminal** Nedir?
  - Kısaca Terminal; kullanıcıların **sadece** klavye ve ekran yardımıyla işletim sistemi veya yazılımları kontrol etmesini veya yönetmesine yardım eden **komut ekranı**dır. 
- `mkdir`
  - Komut ekranımızda `mkdir dizin_adi` komutunu çalıştırır ve argüman olarak **dizin_adi** 'nı yollarsak bulunduğumuz dizinde '**dizin_adi**' adında bir klasör oluşturacaktır.
- `ls`
  - Komut ekranımızda `ls` komutunu çalıştırırsak bize bulunduğumuz dizindeki mevcut dizinleri, dosyaları listeleyecektir.

### Git CLI (Command Line Interface / Komut İstemcisi) Test Edelim
Git'i sistemimize kurduktan sonra, komut istemcimizde `git --version` komutunu çalıştırdığımızda bize sistemimizde yüklü olan Git'in versiyonunu döndürecektir.(Versiyonlarda farklılık olabilir.)
```
git --version
>>> git version 2.26.2
```

# Git'e Giriş
- Bir projemiz olduğunu varsayalım. `cd` komutu sayesinde Git reposu (repository) oluşturmak istediğimiz dizine gidiyoruz.
```
cd Desktop/projedizinim/ 
```
- Ardından `ls` komutu sayesinde dizinimizdeki dosyaları/klasörleri listeleyebiliriz.
```
ls
>>> intro.py
```
- `git init` komutu sayesinde mevcut dizinimizde bir Git Repository'si oluşturabiliriz ve sonrasında boş bir Git Reposu oluşturuldu şeklinde çıktı alacağız.
```
git init
>>> Initialized empty Git repository in /root/Desktop/projedizinim/.git/
```
- Artık **projedizinim** altındaki dosyalar Git ile takip edilebilirler. Fakat **henüz takip edilmiyorlar**. Bunun sebebini aslında yukarıda belirtmiştik. **projedizinim** bizim Working Directory'miz yani çalışma alanımız. Bir dosyanın veya klasörün Working Directory'de olması Git tarafından takip edildiği anlamına gelmez. Bu sebeple dosyalarımızı Staging Area'ya yani Git'in takip ettiği sanal ortama tanıtmamız gerekli. 
- Git reposunu oluşturduğumuz dizin altında `git status` komutunu çalıştıralım. Bize repomuzun durumunu anlatırken aynı zamanda yapmak isteyeceğimiz işlemlerin komutlarını da hatırlatır.
```
git status
>>> On branch master

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        intro.py

nothing added to commit but untracked files present (use "git add" to track)
```
- Git `intro.py` adlı dosyamızı gördü fakat henüz takip etmiyor. Bu sebeple **Untracked files** yani **Takip edilmeyen dosyalar** altında listeledi. Şimdi Git'e bu dosyayı takip et, yani **Staging Area**'ya al dememiz gerekiyor.
- `git add dosya_adi` komutunu kullandığımızda ilgili dosyayı Staging Area'ya ekler ve takip etmeye başlar.
```
git add intro.py # benim repoma eklemek istediğim dosyanın adı intro.py siz de kendi dosyanızı argüman olarak yollamalısınız ^^
``` 
- Şimdi tekrar Git repomuzun durumuna bakalım
```
git status
>>> On branch master

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   intro.py
```
- Git artık dosyalarımızı takip ediyor fakat bu sefer de bize diyorki 'Bu dosyada değişiklikler var ama henüz **commit** edilmemiş'. Bunu da `git commit -m "ilk commit"` komutu sayesinde yapabiliriz. `git commit -m` komutuna argüman olarak yolladığımız string bizim açıklama metnimiz olacaktır. Log kayıtlarında bu şekilde görüntülenecek. Bizim ilk denememiz olduğu için "ilk commit" şeklinde bir açıklama yaptık. 
```
git commit -m "ilk commit"
>>> [master (root-commit) 62cf9b5] ilk commit
 1 file changed, 1 insertion(+)
 create mode 100644 intro.py
```
- Eğer Git'i ilk defa kurduysak bu aşamada git bizden kendimizi tanıtmamızı isteyecektir. Bunun sebebi ise oluşturduğumuz **commitler**i bizim adımıza oluşturacak ve Repository üzerinde kimin değişiklik/güncelleme yaptığının bilgisini tutacaktır. Sistemimizde **tek seferlik** bu konfigürasyonu yapmamız gerekmekte. Gerçek bilgiler olması önemlidir.

```
git config --global user.name "ad soyad"
git config --global user.email "mail adresi"
```
- Konfigürasyonları yaptıktan sonra tekrar commit etmeyi deniyoruz (eğer kendimizi tanıtmamız istenmişse).
```
git commit -m "ilk commit"
>>> [master (root-commit) 62cf9b5] ilk commit
 1 file changed, 1 insertion(+)
 create mode 100644 intro.py
```
#### Not
- Şimdi `intro.py` dosyamızda rastgele değişiklik yapalım ve ardından `git status` komutunu çalıştıralım.
```
git status
>>> On branch master
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   intro.py

no changes added to commit (use "git add" and/or "git commit -a")
```
- Gördüğümüz gibi Git bize hangi dosyalarda değişiklik olmuş, repomuzda neler olmuş bitmiş bunların raporunu vermekte. Buradan anladığımız kadarıyla da `intro.py` değiştirilmiş/güncellenmiş fakat  **henüz commit edilmemiş**. `git status` komutu zaten bize ne yapmamız gerektiğini söylüyor. 
- Şimdi tekrar dosyamızda değişiklik yapalım ve farzedelim ki kodlarımızda her şey yolunda ve bunu prod'a yollayabiliriz. Değişikliklerden sonra tekrar **staging area**'ya ekleyelim ve **commit** edelim.
```
git status
>>> On branch master
On branch master
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   intro.py

no changes added to commit (use "git add" and/or "git commit -a")

git add intro.py

git commit -m "hello world fonksiyonu eklendi"

>>> [master 000d29e] hello world fonksiyonu eklendi
 1 file changed, 1 insertion(+), 1 deletion(-)

```
- Siz de takdir edersiniz ki dosyalarımızın sadece çalışır ve olmasını istediğimiz hallerini commit ettik. **Her zaman dosyalarımızın en son, çalışır hallerini commit etmeliyiz.** Aksi takdirde prod kısmında kodlarımızı güncellediğimizde kötü sonuçlar ile karşılaşmamız kaçınılmazdır.
  
# Branch

## Branch Nedir?
Git commit'lerimizi şimdiye kadar hep lineer bir yapıda tuttuk. Git, commit'leri sadece tek bir dalda değil, birden fazla dal içerebilen bir **ağaç** yapısında tutabilir. Böylece iki ayrı dal (iki ayrı modül) üzerinde çalışabilir ve işimiz bittiğinde bu dalları birleştirerek tek bir dala sahip olabiliriz.
- Default olarak gelen branch'imizin adı `master`'dır.
  
## Peki Neden Branch'lere İhtiyaç Duyarız?
Projelerimiz çoğu zaman tek bir çizgi üzerinde ilerlemez. Eş zamanlı olarak projemizin farklı modülleri üzerinde çalışmak isteyebiliriz, farklı bug fix'leri üzerinde aynı anda çalışmak isteyebiliriz veya web sayfalarımızın birden fazla versiyonunu hazırlayıp test etmek isteyebiliriz. Bu gibi durumlarda projemizin sağlıklı çalışır bir versiyonunu (**branch**) elimizde bulundurmak isteriz. Default olarak gelen `master` branch'imizi **ana** branch'imiz olarak düşünelim. İki farklı dala ayırıp Modül1 üzerinde ve Modül2 üzerinde eş zamanlı olarak çalışabiliriz. Modül2'nin yazımı bittikten sonra Modül1'in yazımını beklemeden yapılan değişiklikleri master branch'e aktarabiliriz. Bu işleme ise `merge` denmektedir. Artık **master branch**'imize Modül2'de dahil olmuş oldu.

- Bir branch oluşturmak için `git branch branch_adi` komutunu oluşturmak istediğimiz branch'in adını argüman olarak yollayarak kullanmamız gerekmektedir. Ardından `git status` ile repomuzun durumunu kontrol edelim ve hangi branch'te olduğumuza bakalım. Demiştik ki ilk git reposu oluşturulduğunda default olarak `master` branch açılır. 
```
git branch haber-listele # haber listele adında branch oluştuduk
git branch musteri-listele # müşteri listele adında branch oluştuduk
git status
>>> On branch master
nothing to commit, working tree clean
```
- Branch değiştirmek için `git checkout branch_adi` komutunu, geçmek istediğimiz branch'in adını argüman olarak vererek kullanabiliriz. Ardından tekrar `git status` ile hangi branch'te olduğumuzu görebiliriz.
```
git checkout haber-listele
>>> Switched to branch 'haber-listele'

git status
>>> On branch haber-listele
nothing to commit, working tree clean
```
- Ardından `haber_getir.py` adında bir dosyayı `haber-listele` adlı branch'ime ekleyeceğim ve sonrasında commit ettikten sonra `git log` komutu ile branch'imde yapılmış olan değişiklikleri göreceğim.
```
cat > haber_listele.py

.def haber_getir():
.        print('Haberler getiriliyor...')
.
.

git add haber_listele.py

git commit -m 'haber listeleme modulu eklendi'
>>> [haber-listele 287f583] haber listeleme modulu eklendi
 1 file changed, 3 insertions(+)
 create mode 100644 haber_listele.py

git log
>>> commit 287f5835608c07fd0569f2d371f5769e76950984 (HEAD -> haber-listele)
Author: mebaysan <menesbaysan@gmail.com>
Date:   Mon May 18 01:44:56 2020 +0300

    haber listeleme modulu eklendi

commit c374c7b8837d2e84b85e41413902898709c1d7b9 (musteri-listele, master)
Author: mebaysan <menesbaysan@gmail.com>
Date:   Mon May 18 01:34:30 2020 +0300

    proje olusturuldu

```
- Şimdi `master` branch'imize geçelim. Ardından da bu branch'te ne gibi değişiklikler olmuş ona bakalım.
```
git checkout master
>>> Switched to branch 'master'

git log
>>> commit c374c7b8837d2e84b85e41413902898709c1d7b9 (HEAD -> master, musteri-listele)
Author: mebaysan <menesbaysan@gmail.com>
Date:   Mon May 18 01:34:30 2020 +0300

    proje olusturuldu

``` 
- Gördüğümüz gibi az önce yaptığımız commit ('haber listeleme modulu eklendi') gözükmüyor. Bunun sebebi ise o commit'in `haber-listele` branch'inde olması. Henüz master branch'i ile **merge** edilmedi.
- Şimdi `musteri-listele` branch'imize geçelim ve biraz değişiklik yapalım ve commit edelim.
```
git checkout musteri-listele
>>> Switched to branch 'musteri-listele'

cat > musteri_getir.py

. def musteri_getir():
.         print('Musteriler getiriliyor..')
.
.

git add musteri_getir.py

git commit -m 'musteri getirme modulu eklendi'
>>> [musteri-listele f4b4584] musteri getirme modulu eklendi
 1 file changed, 3 insertions(+)
 create mode 100644 musteri_getir.py
```
- `haber-listele` branch'inin prod'a çıkmaya hazır olduğunu varsayalım. Ve prod branch'imiz olan master ile birleştirelim (**merge**). İki branch'i **merge** etmek için önce güncelleme yapmak istediğimiz (bu proje için master) branch'e geçelim. Ardından `git merge branch_adi` komutunu merge (birleştirmek) istediğimiz branch adını argüman olarak yollayarak kullanabiliriz. Bu işlemden sonra mevcut (master) branch'imizde ne gibi değişiklikler olmuş görelim.
```
git checkout master # güncelleme yapmak istediğimiz branch'e geçiyoruz
>>> Switched to branch 'master'

git merge haber-listele # birleştirmek istediğimiz branch'i söylüyoruz
>>> Updating c374c7b..287f583
 Fast-forward
 haber_listele.py | 3 +++
 1 file changed, 3 insertions(+)
 create mode 100644 haber_listele.py


git log
>>> commit 287f5835608c07fd0569f2d371f5769e76950984 (HEAD -> master, haber-listele)
Author: mebaysan <menesbaysan@gmail.com>
Date:   Mon May 18 01:44:56 2020 +0300

    haber listeleme modulu eklendi

commit c374c7b8837d2e84b85e41413902898709c1d7b9
Author: mebaysan <menesbaysan@gmail.com>
Date:   Mon May 18 01:34:30 2020 +0300

    proje olusturuldu
```

- Gördüğümüz gibi `master` branch'imizde artık `haber-listele` branch'indeki kodlar da gelmiş bulunmakta.
- `musteri-listele` branch'i ile de işimiz bittiğini varsayalım ve onu da master branch'imiz ile merge edelim. Ardından yapılan değişiklikleri görmek için `git log` komutunu kullanalım.
```
git merge musteri-listele
>>> Merge made by the 'recursive' strategy.
 musteri_getir.py | 3 +++
 1 file changed, 3 insertions(+)
 create mode 100644 musteri_getir.py

git log --oneline

>>> 7b0af94 (HEAD -> master) Merge branch 'musteri-listele'
f4b4584 (musteri-listele) musteri getirme modulu eklendi
287f583 (haber-listele) haber listeleme modulu eklendi
c374c7b proje olusturuldu

```

## Branch'leri Silmek
`musteri-listele` ve `haber-listele` branch'lerini `master` branch'imize taşıdığımza göre artık bu branch'leri silebiliriz. Bunun için `git branch -d branch_adi` komutunu silmek istediğimiz branch'in adını argüman olarak yollayarak kullanabiliriz. Adını yolladığımız branch **silinecektir**. Eğer oluşturduğumuz branch'leri unuttuysak `git branch` komutu ile oluşturduğumuz branchleri ve hangi branch'te olduğumuzu görebiliriz.
```
git branch
>>> haber-listele
    * master # yanında yıldız olan branch, üzerinde çalıştığımız branch'dir
    musteri-listele


git branch -d haber-listele
>>> Deleted branch haber-listele (was 287f583)

git branch -d musteri-listele
>>> Deleted branch musteri-listele (was f4b4584)
```
- Branch'leri silerken endişelenmemize gerek yok. Eğer silmek istediğimiz branch başka bir branch'e merge edilmediyse git bizi uyaracaktır.

# Conflicts (Çakışmalar)
Yukarıda branch'leri **merge** ederken hep farklı dosyalar üzerinde çalıştık ve bunları merge ederken çakışma (conflict) olmadı. Git üç branch'teki üç commit'i birleştirdi. Peki ya **2 farklı branch**'te **aynı dosya/lar** üzerinde değişiklik yapsaydık?

- 2 adet branch oluşturacağım ve ikisinde de musteri.py üzerinde değişiklikler yapacağım.
```
git branch musteri-listele
git branch musteri-getir # 2 adet branch oluşturdum

git checkout musteri-listele # mevcut branch'imi değiştirdim
>>> Switched to branch 'musteri-listele'

cat >> musteri.py
. def musteri_listele():
.        print('Müşteriler listeleniyor..')
.
.

git add musteri.py 

git commit -m 'musteri_listele func eklendi'
>>> [musteri-listele 42632e0] musteri_listele func eklendi
 1 file changed, 3 insertions(+)

```
- `musteri-listele` branch'imizde değişiklikleri yaptık. Şimdi `musteri-getir` branch'inde değişiklik yapalım.
```
git checkout musteri-getir
>>> Switched to branch 'musteri-getir'

cat >> musteri.py
. def musteri_getir():
.        print('Müşteriler getiriliyor..')
.
.

git add musteri.py 

git commit -m 'musteri getir func eklendi'
>>> [musteri-getir 1cf8c90] musteri getir func eklendi
 1 file changed, 3 insertions(+)

```
- 2 farklı branch'de `musteri.py` dosyası üzerinde değişiklik yaptık. Şimdi `master` branch'imize geçip `musteri-listele` branch'ini **merge** edelim.
```
git checkout master
>>> Switched to branch 'master'

git merge musteri-listele
>>> Updating 9a39d7a..42632e0
Fast-forward
 musteri.py | 3 +++
 1 file changed, 3 insertions(+)
```
- `master` ve `musteri-listele` branch'lerini başarılı bir şekilde **merge** ettik. Şimdi de `master` ve `musteri-getir` branch'lerini merge etmeyi **deneyelim**
```
git merge musteri-getir
>>> Auto-merging musteri.py
CONFLICT (content): Merge conflict in musteri.py
Automatic merge failed; fix conflicts and then commit the result.
```
- Evet ilk **conflict (çakışma)** hatamızı aldık. Git bize `musteri.py` içerisinde bir çakışma (conflict) olduğunu, bu sebeple otomatik birleştirme işlemini (merge) gerçekleştiremediğini söylüyor. Dosyamızı açarsak Git bize çakışma hakkında daha detaylı bilgi verecektir. Halihazırda Linux bir sistem kullandığım için `cat` komutu ile dosyamın içeriğini görüntüleyeceğim.
```
cat musteri.py 
>>> print('Musteriler getiriliyor')

<<<<<<< HEAD
def musteri_listele():
        print('Müşteriler listeleniyor..')
=======
def musteri_getir():
        print('Müşteriler getiriliyor..')
>>>>>>> musteri-getir
```
- Gördüğümüz gibi Git bize hangi satırlarda çakışma olduğunun bilgisini veriyor. `<<<` ve `>>>` işaretleri ile gösteriyor.
- `<<<<<<< HEAD` ile `=======` işaretleri arasındaki kısım `master` branch'ine ait mevcut durumu gösteriyor.
- `=======` ile `>>>>>>>` **merge** etmek istediğimiz branch'teki yani `musteri-getir` branch'indeki halini gösteriyor. 
- Bu noktada `<<<<<<<` ve `>>>>>>>` işaretlerini silip dosyamızı istediğimiz, çalışır haline getiriyoruz ve `conflict`'i çözmüş oluyoruz. Ardından tekrar değişen dosyayı `staging area`'ya ekliyoruz ve commit ediyoruz.
```
nano musteri.py # düzenleyeceğim dosyamı açıyorum ve içeriği aşağıdaki gibi düzenliyorum

. def musteri_listele_getir():
.        print('Müşteriler listeleniyor..')
.        print('Müşteriler getiriliyor..')
.
.

git add musteri.py # dosyamızı staging area'ya ekliyoruz
git commit -m 'conflict cozuldu' # commit ediyoruz
>>> [master 34463c6] conflict cozuldu
git status # durumu gözden geçiriyoruz
>>> On branch master
nothing to commit, working tree clean
```

# Gitignore (Dosyaları Yok Saymak)
Git otomatik olarak repo klasörümüz içerisindeki **bütün dosyaları** takip eder. Peki biz local'de farklı çalışan dosyaları uzak sunucuya göndermek istemezsek? Bu noktada da `.gitignore` adlı dosyayı kullanıyoruz. Başında '`.`' olmasının sebebi bu dosyanın gizli bir dosya olmasından kaynaklanmaktadır. Bu dosya içerisine yazılan **dosya/dizin** isimleri Git tarafından takip edilmez ve repolarımıza eklenmez. Bu dosya içerisinde `regex` kullanmamız mümkündür.
- Temiz bir repo oluşturalım ve içine `global.py`,`local.py`,`deneme.txt` ve `deneme2.txt` adında dosyaları ekleyelim. Ardından `.gitignore` adında bir dosya oluşturup içerisine `local.py` ve `*txt` satırlarını ekleyip `git status` ile repomuzun durumuna bakalım.
```
git status # dosyalarımı ilk eklediğim anda repo durumu sorguluyorum
>>> On branch master

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        deneme1.txt
        deneme2.txt
        global.py
        local.py

nothing added to commit but untracked files present (use "git add" to track)

touch .gitignore # dosyamı oluşturdum
vim .gitignore
. local.py
. *.txt
.
.

git status
>>> On branch master

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        .gitignore
        global.py

nothing added to commit but untracked files present (use "git add" to track)
```
- Gördüğümüz gibi Git working directory'de sadece `.gitignore` ve `global.py` dosyalarını görmekte. `.gitignore` dosyası içine yazdığımız `local.py` satırı o isimdeki dosyanın Git tarafından takip edilmemesini sağladı. Aynı şekilde `*.txt` satırı ise sonu '**.txt**' ile bütün dosyaların takip edilmemesini sağladı.


# Uzak Sunucu
Şimdiye kadar hep local makinamızda repolar oluşturduk ve işlemler gerçekleştirdik. Git'in kullanıldığı uzak sunucular vardır ve buralarda birden fazla geliştirici bir proje üzerinde aynı anda çalışmalar yapabilir. **Git != GitHub || GİtLab** . Bu uzak sunuculardan en çok bilinenleri:
- [GitHub](https://github.com/)
- [GitLab](https://about.gitlab.com/)
- [Açık Kaynak Kod Platformu](https://kod.acikkaynak.gov.tr/explore/projects/starred) (Belki de henüz pek bilinmese de yerli ve milli bu hizmeti de listeye eklemek istedim)

## Push - Pull Nedir?
Yerel (local) makinadaki commit'leri uzak sunucuya göndermek (itmek) demektir. `git push` komutu sayesinde local commit'lerimizi uzak sunucuya gönderebiliriz. Aynı zamanda uzak sunucudaki commit'leri yerel (local) makinamıza çekmek için ise `git pull` komutunu kullanmamız gerekiyor. 
### Ufak Bir Not
Herhangi bir uzak sunucu ile çalışırken  commit'leri `push` etmeden önce `pull` etmemiz gerekmektedir. Bunun sebebi ise farklı geliştiricilerle çalışıyorsak veya birden fazla commit varsa, commit'lerin sırasının bozulmaması gerekmektedir.

Git repomuza birden fazla uzak sunucu ekleyebiliriz. İlk eklenen uzak sunucu her zaman `origin` olarak adlandırılır. `git remote` komutuyla repomuza bir uzak sunucu eklerken isimlendirmemiz mümkündür.
```
git remote add uzak-sunucu1 UZAK_SUNUCU_ADRESI
```
Ve uzak-sunucu1 adındaki uzak sunucumuza repomuzu tanımladık. Bu uzak sunucuya dosyalarımızı,commit'lerimizi `push` ederken şu komutu çalıştırabiliriz. `git push uzak-sunucu1 master` Şu manaya gelmektedir: `git push` -> localdeki commit'leri gönder, `uzak-sunucu1` -> uzak sunucu adı / hangi uzak sunucuya gönderileceği, `master` uzak-sunucu1'deki master branch'ine push et. <br>

Peki eklediğimiz uzak sunucuları nasıl göreceğiz? Bunun için ise `git remote` komutunu kullanmamız yeterlidir.
```
git remote
>>> akkp # açık kaynak kod platformu
origin # ilk eklediğimiz uzak sunucu (github)
```

## GitHub ile devam edelim
GitHub üzerinden bir repo açtığımızı varsayalım. Ve repo'muzu ilk açtığında bizi böyle bir ekran karşılayacaktır.
![GitHubNewRepo](./assets/1.png)

Eğer hali hazırda bir Git repomuz yok ise ilk bloktaki kodlara tabi olacağız, kodlarımız başka bir uzak sunucuda ise 3. bloğa tabi olacağız. Local'de bir repomuz var fakat henüz bunu bir uzak sunucuya atmamışsak 2. bloğa tabi olacağız ki ben burada 2. blok üzerinden çalışmalarıma devam edeceğim. 
```
git remote add origin https://github.com/mebaysan/Deneme.git
git push -u origin master
```
Şimdi bu komutları inceleyecek olursak; <br>
`git remote add origin` -> **origin** adında bir uzak sunucuyu repomuza ekler, `https://github.com/mebaysan/Deneme.git` -> uzak sunucumuzun adresidir.  <br>
`git push -u` -> local'deki commit'leri push et, `origin` -> repoma tanıttığım **origin** isimli sunucuya, `master` -> origin isimli sunucumdaki **master branch**'ine. <br>
`git remote add SUNUCU_ADI SUNUCU_ADRESI` -> bu komut yapısı ile repo'ya uzak sunucu ekleyebiliriz <br>
`git push -u SUNUCU_ADI BRANCH_ADI` -> bu komut yapısı ile istediğimiz uzak sunucuda istediğimiz branch'e local commit'leri push edebiliriz.

## Clone Nedir
![Git Clone](./assets/2.png)

Projelerimizi uzak sunucuya `push` ettikten sonra, bu repoyu o uzak sunucudan herhangi bir başka sisteme çekmek için `git clone repo_adresi` komutunu clone'lamak istediğimiz repo'nun adresini argüman olarak yollayarak kullanabiliriz. Ardından uzak sunucudaki repo'muz yerel (local) sistemimize kopyalanmış olacaktır. Bu işleme **clone** denmektedir. GitHub için **clone** adresine repo'muzun detayına girdikten sonra sağ üst köşeden erişebiliriz. Ardından şu şekilde komutu kullanabiliriz.
```
git clone https://github.com/mebaysan/TemelGitElKitabi.git
>>> Cloning into 'TemelGitElKitabi'...
remote: Enumerating objects: 17, done.
remote: Counting objects: 100% (17/17), done.
remote: Compressing objects: 100% (12/12), done.
remote: Total 17 (delta 4), reused 17 (delta 4), pack-reused 0
Receiving objects: 100% (17/17), 63.19 KiB | 681.00 KiB/s, done.
Resolving deltas: 100% (4/4), done.
```
## Referans: https://github.com/mebaysan/TemelGitElKitabi
