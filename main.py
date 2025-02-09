from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# Initialize ChatGPT model
model_local = ChatOpenAI(model_name="gpt-4")

# 1. Split data into chunks
resume = """
# Sam Siggs
## Software Engineer

### Profile
I am a software developer with over 7 years of experience in New Zealand,
creating solutions that drive revenue, streamline complexity, enhance security,
and boost user engagement. I specialize in breaking down complex technical systems for non-technical stakeholders,
managing multiple priorities under strict deadlines, and maintaining high performance standards.
I focus on web development, integrations, security, automation, database management,
and DevOps, including Docker and automated pipelines.
My greatest strength is my ability to innovate and adapt to new technologies,
developing cutting-edge solutions to stay ahead in a rapidly evolving industry.

### Work Experience
#### Security Software Developer
#### InPhySec Security - Fujitsu Cyber Security 
#### 2023-Current
- Improved visibility into a large amount of our services, allowing for earlier detection and improved customer experience
- Improved communication between clients during the CrowdStrike / Microsoft Windows outage allowing for faster information transfer assisting them through their outage
- Improved virtual machine and container reliability and security through scheduled dependency auditing and continuous upgrading
- Completed major and minor upgrades of various services including Gitlab and Docker Engine increasing security and reliability 
- Integrate two-way synchronization between various security platforms including Microsoft Defender, Falcon and Zendesk, streamlining SOC operational activity and increasing efficiency
- Overhauled Kubernetes infrastructure using OpenTofu, resolving platform vulnerabilities, consolidating technologies like HAProxy, improving maintainability, enhancing security, and streamlining operations
- Improved communication between stakeholders and developers by migrating collaboration data from a standalone, segregated platform to a unified platform accessible to both sides, enhancing transparency and efficiency

#### Integrations Developer
#### Overdose Digital
#### 2023-2024
- Overhauled search functionality for Asics, significantly improving user experience, increasing sales by approximately 5%, improving SEO, and optimizing mobile viewing.
- Implemented bundled product offerings for Animates, boosting sales by an estimated 4%, enhancing customer value, accelerating sales of slower-moving stock, simplifying purchasing processes.
- Streamlined order integration for Vine Online, reducing incorrect orders by about 30%, improving stock management, increasing scalability.
- Updated the VAPO AU website to comply with Australia's new laws, enhancing regulatory compliance.
- Implemented security features and updates for various clients including Barkers, Max and Bobux, including rate limiting on login pages, Magento security updates, New Relic tracking for attack detection and reCAPTCHA on forms, greatly enhancing security and reducing the risk of data breaches by 40%.
- Overhauled social login features for Timberland, Birkenstock, and Daiwa, greatly enhancing security and reducing the risk of data breaches by 80%.
- Updated Advintage website to align New Zealand's liquor sale and advertising laws, improving compliance and avoiding potential fines of up to $10,0000

#### Web Developer
#### PB Tech
#### 2022-2023
- Developed a health and safety system, improving compliance, enhancing employee safety, better incident management, supporting regular reporting, and reducing workplace incidents by approximately 25%.
- Created an AI microservice, reducing integration costs by 10%, increasing scalability, and enhancing data insights.
- Implemented Multi-Factor Authentication, reducing the risk of data breaches by 50% and curbing illegal activity.
- Built a development environment using Docker, improving onboarding efficiency, development efficiency, and scalability for development tools, while enhancing security.
- Engineered automated pipelines, streamlining the development process, increasing release frequency by 20%, reducing human errors by 50%, and enhancing scalability and security through code analysis.

#### Contract Software Developer
#### Kiwischools
#### 2021-2023
- Rebuilt the mobile app, improving user experience, increasing adoption and usage by 15%, and enhancing stability and scalability.
- Enhanced WordPress sites, significantly improving performance, security, and SEO, facilitating better integration with the mobile application, and supporting third-party integrations.
- Successfully overhauled and migrated MariaDB server, enhancing system performance and reliability by 25%

#### Contract Software Developer
#### Onion CRM
#### 2020-2021
- Automated lead tracking for Cognition Education and Boston Wardrobes, enhancing lead management, response times, and sales, while improving lead segmentation, targeting, and reporting.
- Automated mailing address synchronization for Copyright NZ's client base, improving data accuracy and consistency, enhancing communication, and reducing maintenance costs.
- Implemented automated project analytics for Cognition Education, enhancing data collection and analytics, improving reporting, risk management, and resource allocation.

#### Contract Software Developer
#### SteelPencil
#### 2019-2020
- Designed software solutions to enhance access to information and reveal previously inaccessible data, improving overall system usability and access.
- Designed and implemented new software to accelerate data collection and manipulation, significantly boosting productivity and data handling efficiency.
- Automated all necessary emails, reducing the workload of employees by 20% and increasing operational efficiency.

### Skills
#### Version Control Systems
- Git
- BitBucket
- GitLab
- GitHub

#### Programming Languages
- PHP
- JavaScript
- Python
- Golang
- Java

#### Frameworks and Libraries
- Symfony
- Bootstrap
- Laravel
- React
- React Native
- FastAPI
- NodeJS
- API Platform
- jQuery
- Flask
- HTMX
- Tailwind

#### Security Platforms
- Defender
- Sentinel
- Falcon

#### Databases
- MySQL
- MariaDB
- MSSQL

#### E-Commerce Platforms
- VTEX
- BigCommerce
- Magento 2

#### CI/CD
- GitLab CI/CD
- GitHub Actions
- Jenkins
- BitBucket Pipelines

#### Operating Systems
- Linux
- Windows
- MacOS

#### Cloud Services and Platforms
- AWS
- Docker
- Portainer
- Heroku
- Digital Ocean
- Proxmox
- VM Ware
- Terraform
- OpenTofu

#### Collaboration Platforms
- Teams
- Google Chat
- Slack

#### CRM and Business Software
- Zoho
- Zendesk

#### Project Management
- Jira
- Wrike

#### Query Languages
- GraphQL
- REST

#### AI and Machine Learning
- OpenAI
- Ollama

#### Content Management Systems
- WordPress
- Unleashed Software

#### Observability and Monitoring
- Elastic
- New Relic

"""
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=7500, chunk_overlap=100
)
doc_splits = text_splitter.split_text(resume)

# 2. Convert documents to Embeddings and store them
vectorstore = Chroma.from_texts(
    texts=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

# 3. Prompt the model and output the result
after_rag_template = """
Cover Letter Creator focuses on generating the main content of the cover letter without any headings or personal contact information.
It directly crafts a narrative that highlights the user's skills and experiences relevant to the job description, maintaining a formal tone throughout.
The GPT is designed to start the cover letter with an introduction relevant to the job role and company,
and if any additional information is needed for accuracy, it will request it from the user.
Do not hallucinate or make up things that aren't in my Resume
Resume: {resume}
Job Description: {job_description}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"resume": retriever, "job_description": RunnablePassthrough()} | after_rag_prompt | model_local | StrOutputParser()
)

job_description = """
   {"Advertiser:43189181":{"__typename":"Advertiser","id":"43189181","name({\"locale\":\"en-NZ\"})":"PHQ","isVerified":true,"registrationDate":{"__typename":"SeekDateTime","dateTimeUtc":"2019-08-07T09:19:22.267Z"}},"JobProductBranding:c30de186-242b-d561-24b5-39e81f12ae7e.1":{"__typename":"JobProductBranding","id":"c30de186-242b-d561-24b5-39e81f12ae7e.1","cover":{"__typename":"JobProductBrandingImage","url":"https:\u002F\u002Fimage-service-cdn.seek.com.au\u002F3943839e715a12719d7d0887e0f8563dd001b6e6\u002Fe72391e6f0492aa5c0258cca8437b88cb3102728"},"cover({\"isThumbnail\":true})":{"__typename":"JobProductBrandingImage","url":"https:\u002F\u002Fimage-service-cdn.seek.com.au\u002F3943839e715a12719d7d0887e0f8563dd001b6e6\u002F81fe2c0a262aa407b15517bb5c31a4bf1889dcb3"},"logo":{"__typename":"JobProductBrandingImage","url":"https:\u002F\u002Fimage-service-cdn.seek.com.au\u002Ffcdbc877c0713f559d4c28cd31d098aebd866ea8\u002Ff3c5292cec0e05e4272d9bf9146f390d366481d0"}},"ROOT_QUERY":{"__typename":"Query","jobDetails:{\"id\":\"81769096\"}":{"__typename":"JobDetails","job":{"__typename":"Job","sourceZone":"anz-2","tracking":{"__typename":"JobTracking","adProductType":null,"classificationInfo":{"__typename":"JobTrackingClassificationInfo","classificationId":"6281","classification":"Information & Communication Technology","subClassificationId":"6287","subClassification":"Developers\u002FProgrammers"},"hasRoleRequirements":true,"isPrivateAdvertiser":false,"locationInfo":{"__typename":"JobTrackingLocationInfo","area":"Auckland Central","location":"Auckland","locationIds":["30172"]},"workTypeIds":"242","postedTime":"5d ago"},"id":"81769096","title":"Senior Backend Developer","phoneNumber":null,"isExpired":false,"expiresAt":{"__typename":"SeekDateTime","dateTimeUtc":"2025-03-05T12:59:59.999Z"},"isLinkOut":false,"contactMatches":[],"isVerified":true,"abstract":"PHQ are looking for a Senior Backend Developer to bring their problem-solving skills to our Auckland studio.","content({\"platform\":\"WEB\"})":"\u003Cp\u003E\u003Cstrong\u003EAbout us\u003C\u002Fstrong\u003E\u003C\u002Fp\u003E\u003Cp\u003EPHQ (https:\u002F\u002Fphq.nz) is an independent, digitally led creative agency based in Auckland, founded by the Phantom London studio (https:\u002F\u002Fphantom.land\u002F). Our very essence is to shape shift - to adapt and evolve, collaborating with brands that challenge and endlessly inspire us.  We transform challenges of all shapes and sizes into inventive, engaging and performance driven solutions. For us, every project is a unique opportunity to lead the industry and deliver groundbreaking results.\u003C\u002Fp\u003E\u003Cp\u003EPut simply, we want to do incredible work that we’re proud of and make our clients love us. Where there’s an innovative solution to be had or simple amends to complete, we will tackle it even better than anyone could have imagined. \u003C\u002Fp\u003E\u003Cp\u003E\u003Cstrong\u003EAbout the role\u003C\u002Fstrong\u003E\u003C\u002Fp\u003E\u003Cp\u003EAs a Senior Backend Developer, you will collaborate with other Developers, Designers and Producers on technical solutions that align closely with a creative vision (web apps &amp; games, cloud architecture, devops, Gen-AI, creative tools &amp; installations etc). \u003C\u002Fp\u003E\u003Cp\u003E\u003Cstrong\u003EAbout you\u003C\u002Fstrong\u003E\u003C\u002Fp\u003E\u003Cp\u003EWe are looking for an enthusiastic Senior Backend Developer who has strong experience in web development &amp; related technologies. We are looking for individuals who tick these boxes: \u003C\u002Fp\u003E\u003Cp\u003E\u003Cstrong\u003EYour credentials: \u003C\u002Fstrong\u003E\u003C\u002Fp\u003E\u003Cul\u003E\u003Cli\u003EMinimum of 6 years development experience in a commercial or agency environment. \u003C\u002Fli\u003E\u003Cli\u003EExpert knowledge in backend programming languages and methodologies (Python, PHP, Typescript etc)\u003C\u002Fli\u003E\u003Cli\u003EA strong understanding of Git &amp; confidence with command line tools\u003C\u002Fli\u003E\u003Cli\u003EA strong understanding of secure web development practises (e.g. OWASP)\u003C\u002Fli\u003E\u003Cli\u003EA good understanding of frontend web technologies (HTML5, CSS3, JavaScript)\u003C\u002Fli\u003E\u003Cli\u003EExperience with giving\u002Freceiving code reviews\u003C\u002Fli\u003E\u003Cli\u003EExcellent written and verbal communication skills demonstrated through detailed presentation on technical concepts. \u003C\u002Fli\u003E\u003Cli\u003EExcellent documentation skills including creating meaningful technical specs\u003C\u002Fli\u003E\u003Cli\u003EStrong ability to work on simultaneous projects and meet deadlines\u003C\u002Fli\u003E\u003C\u002Ful\u003E\u003Cp\u003E\u003Cstrong\u003EBonus points:\u003C\u002Fstrong\u003E\u003C\u002Fp\u003E\u003Cul\u003E\u003Cli\u003EDegree in Computer Science or related technical discipline\u003C\u002Fli\u003E\u003Cli\u003EYour own website, showcasing your own work and interests\u003C\u002Fli\u003E\u003Cli\u003EOpen source \u002F community contributions (e.g. on GitHub)\u003C\u002Fli\u003E\u003Cli\u003EExperience with the following:\u003C\u002Fli\u003E\u003Cli\u003ECMS frameworks (e.g. Wagtail\u002FDjango, Drupal)\u003C\u002Fli\u003E\u003Cli\u003ESQL databases (e.g. Postgres, MySQL)\u003C\u002Fli\u003E\u003Cli\u003EFrontend frameworks (e.g. React, Angular)\u003C\u002Fli\u003E\u003Cli\u003EAutomated testing (unit testing etc)\u003C\u002Fli\u003E\u003Cli\u003EDevops tools &amp; internet technologies (e.g. Linux, Docker, CI\u002FCD, Terraform, HTTP, DNS etc) for both development and production environments\u003C\u002Fli\u003E\u003Cli\u003ECloud platforms - Google Cloud, AWS, Azure\u003C\u002Fli\u003E\u003C\u002Ful\u003E\u003Cp\u003E\u003Cstrong\u003EInclusivity at PHQ\u003C\u002Fstrong\u003E\u003C\u002Fp\u003E\u003Cp\u003EWe are an equal opportunity employer and believe our environment, ideas and work can only be its best if our team is representative of all the incredible diversity of thought that exists in the world around us.\u003Cbr \u002F\u003E\u003Cbr \u002F\u003EAll PHQ team members are welcome in our team without regard to ethnicity, religion, national origin, age, disability, sexual orientation or gender identity.\u003C\u002Fp\u003E","status":"Active","listedAt":{"__typename":"SeekDateTime","label({\"context\":\"JOB_POSTED\",\"length\":\"SHORT\",\"locale\":\"en-NZ\",\"timezone\":\"UTC\"})":"5d ago","dateTimeUtc":"2025-02-03T03:34:54.471Z"},"salary":null,"shareLink({\"locale\":\"en-NZ\",\"platform\":\"WEB\",\"zone\":\"anz-2\"})":"https:\u002F\u002Fwww.seek.co.nz\u002Fjob\u002F81769096?tracking=SHR-WEB-SharedJob-anz-2","workTypes":{"__typename":"JobWorkTypes","label({\"locale\":\"en-NZ\"})":"Full time"},"advertiser":{"__ref":"Advertiser:43189181"},"location":{"__typename":"LocationInfo","label({\"locale\":\"en-NZ\",\"type\":\"LONG\"})":"Parnell, Auckland"},"classifications":[{"__typename":"ClassificationInfo","label({\"languageCode\":\"en\"})":"Developers\u002FProgrammers (Information & Communication Technology)"}],"products":{"__typename":"JobProducts","branding":{"__ref":"JobProductBranding:c30de186-242b-d561-24b5-39e81f12ae7e.1"},"bullets":["Exciting global clients & project opportunities","Fun & social team environment","Flexible working opportunities & more benefits"],"questionnaire":{"__typename":"JobQuestionnaire","questions":["How many years' experience do you have as a Backend Developer?","Which of the following statements best describes your right to work in New Zealand?","Which of the following programming languages are you experienced in?","What's your expected annual base salary?"]},"video":{"__typename":"VideoProduct","url":"https:\u002F\u002Fwww.youtube.com\u002Fembed\u002FPgewQwkxE2I","position":"BOTTOM"}}},"companyProfile({\"zone\":\"anz-2\"})":null,"companySearchUrl({\"languageCode\":\"en\",\"zone\":\"anz-2\"})":"https:\u002F\u002Fwww.seek.co.nz\u002FPHQ-jobs\u002Fat-this-company","companyTags":[],"restrictedApplication({\"countryCode\":\"NZ\"})":{"__typename":"JobDetailsRestrictedApplication","label({\"locale\":\"en-NZ\"})":null},"sourcr":null,"learningInsights({\"locale\":\"en-NZ\",\"platform\":\"WEB\",\"zone\":\"anz-2\"})":{"__typename":"LearningInsights","analytics":{"title":"Senior Backend Developer","landingPage":"CA Role Salary","resultType":"CareerAdviceSalaryTeaserSearch:backend-developer","entity":"career","encoded":"title:Senior Backend Developer;landingPage:CA Role Salary;resultType:CareerAdviceSalaryTeaserSearch:backend-developer;entity:career"},"content":"\u003Cstyle\u003E\n  \u002F* capsize font, don't change *\u002F\n  .capsize-heading4 {\n    font-size: 20px;\n    line-height: 24.66px;\n    display: block;\n    font-weight: 500;\n  }\n\n  .capsize-heading4::before {\n    content: '';\n    margin-bottom: -0.225em;\n    display: table;\n  }\n\n  .capsize-heading4::after {\n    content: '';\n    margin-top: -0.225em;\n    display: table;\n  }\n\n  .capsize-standardText {\n    font-size: 16px;\n    line-height: 24.528px;\n    display: block;\n  }\n\n  .capsize-standardText::before {\n    content: '';\n    margin-bottom: -0.375em;\n    display: table;\n  }\n\n  .capsize-standardText::after {\n    content: '';\n    margin-top: -0.375em;\n    display: table;\n  }\n\n  @media only screen and (min-width: 740px) {\n    .capsize-heading3 {\n      font-size: 24px;\n      line-height: 29.792px;\n    }\n\n    .capsize-heading3::before {\n      content: '';\n      margin-bottom: -0.2292em;\n      display: table;\n    }\n\n    .capsize-heading3::after {\n      content: '';\n      margin-top: -0.2292em;\n      display: table;\n    }\n  }\n  \u002F* end of capsize *\u002F\n\n  \u002F* LMIS css start here*\u002F\n  .lmis-root {\n    margin: -32px;\n    padding: 32px;\n    font-family: SeekSans, 'SeekSans Fallback', Arial, sans-serif;\n    background: #beeff3;\n    border-radius: 16px;\n    color: #2e3849;\n  }\n\n  .lmis-title {\n    margin-bottom: 8px;\n  }\n\n  .lmis-cta {\n    min-height: 48px;\n    display: flex;\n    align-items: center;\n    color: #2e3849;\n    text-decoration: none;\n  }\n\n  .lmis-cta-text {\n    margin-right: 4px;\n    font-weight: 500;\n  }\n\n  .lmis-teaser-image {\n    max-width: 96px;\n  }\n\n  @media only screen and (min-width: 992px) {\n    .lmis-root {\n      margin: -48px;\n    }\n\n    .lmis-wrapper {\n      display: flex;\n      flex-direction: row-reverse;\n      justify-content: space-between;\n      align-items: center;\n    }\n  }\n\u003C\u002Fstyle\u003E\n\n\u003Cdiv class=\"lmis-root\"\u003E\n  \u003Cdiv class=\"lmis-wrapper\"\u003E\n    \u003Cdiv class=\"lmis-teaser-image\"\u003E\n      \u003Cimg\n        src=\"https:\u002F\u002Fcdn.seeklearning.com.au\u002Fmedia\u002Fimages\u002Flmis\u002Fgirl_comparing_salaries.svg\"\n        alt=\"salary teaser image\"\n      \u002F\u003E\n    \u003C\u002Fdiv\u003E\n    \u003Cdiv class=\"lmis-content\"\u003E\n      \u003Cdiv class=\"capsize-heading4 lmis-title\"\u003EWhat can I earn as a Backend Developer\u003C\u002Fdiv\u003E\n      \u003Ca\n        class=\"lmis-cta\"\n        href=\"https:\u002F\u002Fwww.seek.co.nz\u002Fcareer-advice\u002Frole\u002Fbackend-developer\u002Fsalary?campaigncode=lrn:skj:sklm:cg:jbd:alpha\"\n        target=\"_blank\"\n      \u003E\n        \u003Cspan class=\"capsize-standardText lmis-cta-text\"\u003ESee more detailed salary information\u003C\u002Fspan\u003E\n        \u003Cimg\n          src=\"https:\u002F\u002Fcdn.seeklearning.com.au\u002Fmedia\u002Fimages\u002Flmis\u002Farrow_right.svg\"\n          alt=\"salary teaser link arrow\"\n        \u002F\u003E\n      \u003C\u002Fa\u003E\n    \u003C\u002Fdiv\u003E\n  \u003C\u002Fdiv\u003E\n\u003C\u002Fdiv\u003E\n"},"gfjInfo":{"__typename":"GFJInfo","location":{"__typename":"GFJLocation","countryCode":"NZ","country({\"locale\":\"en-NZ\"})":"New Zealand","suburb({\"locale\":\"en-NZ\"})":"Parnell","region({\"locale\":\"en-NZ\"})":null,"state({\"locale\":\"en-NZ\"})":null,"postcode":"1149"},"workTypes":{"__typename":"GFJWorkTypes","label":["FULL_TIME"]}},"seoInfo":{"__typename":"SEOInfo","normalisedRoleTitle":"Backend Developer","workType":"242","classification":["6281"],"subClassification":["6287"],"where({\"zone\":\"anz-2\"})":"Parnell Auckland"}}}}
"""
result = after_rag_chain.invoke(job_description)
print()
print(result)
